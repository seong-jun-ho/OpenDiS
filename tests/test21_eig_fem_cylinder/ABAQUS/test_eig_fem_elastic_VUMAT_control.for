!DIR$ FREEFORM
!=======================================================================
!  test21 용 DDM VUMAT  (2026-11-16)
!
!  DD <-> FEM 고리의 FEM 쪽 절반:
!
!    (1) OpenDiS 가 쓴 Data_segs.txt 를 읽어 eigenstrain 을 실제로 뺀다
!          sigma = sigma_old + C : (d_eps - d_eps_p)
!    (2) 적분점 응력을 모아 FEMSTRESS 를 쓴다
!          -> OpenDiS 의 ReadRemoteStress 가 그걸 읽는다
!
!  규약은 우리 ParaDiS DDM VUMAT (VUMAT_DDM_multiple_DDcycle.for) 에서 가져왔다:
!
!    common_omega = t_step * PI * EffSlip^2 / burgers      (:1085)
!    travel       = |r1n + r2n - r1o - r2o| / 2            (:1134-1137)
!    temp_omega   = common_omega / travel                  (:1139)
!    d_eps_p     += de_p / temp_omega                      (:1187-1192)
!  좌표는 b 단위이므로 burgers = 1.0 (원본 :988 과 같다).
!
!  [원본과 다른 점 — 일부러 그렇게 했다]
!  원본은 한 DD 스텝 안의 ABAQUS 증분마다 세그먼트 위치를 보간해 활성부피를
!  움직여 가며 조금씩 더한다. 여기서는 활성부피를 sweep 중간 위치에 고정하고
!  증분의 시간 몫 (dt/t_step) 으로 나눠 더한다. 그러면 한 DD 스텝에 더해진
!  총량이 증분 크기와 무관하게 정확히 같아져 검증하기 쉽다.
!=======================================================================

subroutine vumat(&
    nblock, ndir, nshr, nstatev, nfieldv, nprops, lanneal, &
    stepTime, totalTime, dt, cmname, coordMp, charLength, &
    props, density, strainInc, relSpinInc, &
    tempOld, stretchOld, defgradOld, fieldOld, &
    stressOld, stateOld, enerInternOld, enerInelasOld, &
    tempNew, stretchNew, defgradNew, fieldNew, &
    stressNew, stateNew, enerInternNew, enerInelasNew)

    include 'vaba_param.inc'

    dimension props(nprops), density(nblock), &
        coordMp(nblock,*), charLength(*), strainInc(nblock,ndir+nshr), &
        relSpinInc(*), tempOld(*), stretchOld(*), defgradOld(*), &
        fieldOld(*), stressOld(nblock,ndir+nshr), &
        stateOld(nblock,nstatev), enerInternOld(nblock), &
        enerInelasOld(nblock), tempNew(*), stretchNew(*), &
        defgradNew(*), fieldNew(*), stressNew(nblock,ndir+nshr), &
        stateNew(nblock,nstatev), enerInternNew(nblock), &
        enerInelasNew(nblock)

    character*80 cmname
    parameter (zero=0.0d0, one=1.0d0)

!   ---- DDM: 세그먼트 (Data_segs.txt 에서) -----------------------------
    INTEGER, PARAMETER :: nseg_max = 20000
    DOUBLE PRECISION :: sg_old(nseg_max,6), sg_new(nseg_max,6)
    DOUBLE PRECISION :: sg_dep(nseg_max,6), sg_trv(nseg_max)
    DOUBLE PRECISION :: ddm_tstep, ddm_eff
    INTEGER :: sg_id(nseg_max,2), n_seg_ddm
    COMMON /ddmsegd/ sg_old, sg_new, sg_dep, sg_trv, ddm_tstep, ddm_eff
    COMMON /ddmsegi/ sg_id, n_seg_ddm

!   ---- DDM: 적분점 보관 (FEMSTRESS 용) --------------------------------
    INTEGER, PARAMETER :: nmp_max = 300000
    DOUBLE PRECISION :: mp_xyz(nmp_max,3), mp_sig(nmp_max,6)
    INTEGER :: n_mp_ddm
    COMMON /ddmmpd/ mp_xyz, mp_sig
    COMMON /ddmmpi/ n_mp_ddm

    DOUBLE PRECISION :: dep(6), pmid(3), a(3), v(3), w(3)
    DOUBLE PRECISION :: alen, proj, rperp, frac, omega, PI
    INTEGER :: i, j, k, ii

    PI = 3.14159265358979d0

    E  = props(1)
    NU = props(2)
    G      = E / (2.0d0 * (1.0d0 + NU))
    LAMBDA = E * NU / ((1.0d0 + NU)*(1.0d0 - 2.0d0 * NU))

!   de_p/temp_omega 는 **변형률이 아니라 변형률 속도** [1/s] 다.
!   원본은 거기에 증분 시간 DT 를 곱해 쓴다:
!       dstranp33(1,1) = STATEOLD(iBlock,15) * DT     (원본 :350)
!   test21 은 핸드셰이크가 증분마다 걸려 **DD 스텝당 ABAQUS 증분이 하나**다.
!   그래서 그 DD 스텝의 sweep 전체가 한 번에 들어가도록 t_step 을 곱한다
!   (증분이 여러 개로 쪼개지면 여기를 DT 로 바꾸고 활성부피도 움직여야 한다).
    frac = ddm_tstep

    do i = 1, nblock

!       ---- eigenstrain 누적: 이 적분점을 쓸고 간 세그먼트들 -----------
        do j = 1, 6
            dep(j) = zero
        end do

        IF (n_seg_ddm .GT. 0 .AND. ddm_eff .GT. 0.0d0) THEN
            do k = 1, n_seg_ddm
                IF (sg_trv(k) .LE. 0.0d0) CYCLE

!               sweep 중간 위치의 세그먼트 (a -> a+v)
                do ii = 1, 3
                    a(ii) = 0.5d0*(sg_old(k,ii)   + sg_new(k,ii))
                    v(ii) = 0.5d0*(sg_old(k,ii+3) + sg_new(k,ii+3)) - a(ii)
                    w(ii) = coordMp(i,ii) - a(ii)
                end do

                alen = sqrt(v(1)*v(1) + v(2)*v(2) + v(3)*v(3))
                IF (alen .GT. 0.0d0) THEN
                    proj = (w(1)*v(1) + w(2)*v(2) + w(3)*v(3)) / (alen*alen)
                    IF (proj .LT. 0.0d0) proj = 0.0d0
                    IF (proj .GT. 1.0d0) proj = 1.0d0
                ELSE
                    proj = 0.0d0
                END IF

                rperp = 0.0d0
                do ii = 1, 3
                    rperp = rperp + (w(ii) - proj*v(ii))**2
                end do
                rperp = sqrt(rperp)

!               활성부피: 세그먼트에서 EffSlip 이내인 원통
                IF (rperp .LE. ddm_eff) THEN
!                   omega = common_omega / travel      (원본 :1139)
                    omega = ddm_tstep * PI * ddm_eff * ddm_eff / sg_trv(k)
                    IF (omega .GT. 0.0d0) THEN
                        do j = 1, 6
                            dep(j) = dep(j) + frac * sg_dep(k,j) / omega
                        end do
                    END IF
                END IF
            end do
        END IF

!       ---- 응력 갱신:  sigma = sigma_old + C : (d_eps - d_eps_p) ------
        trace_eps = (strainInc(i,1) - dep(1)) &
                  + (strainInc(i,2) - dep(2)) &
                  + (strainInc(i,3) - dep(3))

        stressNew(i,1) = stressOld(i,1) + LAMBDA*trace_eps &
                       + 2.0d0*G*(strainInc(i,1) - dep(1))
        stressNew(i,2) = stressOld(i,2) + LAMBDA*trace_eps &
                       + 2.0d0*G*(strainInc(i,2) - dep(2))
        stressNew(i,3) = stressOld(i,3) + LAMBDA*trace_eps &
                       + 2.0d0*G*(strainInc(i,3) - dep(3))
        stressNew(i,4) = stressOld(i,4) + 2.0d0*G*(strainInc(i,4) - dep(4))
        if (nshr .ge. 2) stressNew(i,5) = stressOld(i,5) &
                       + 2.0d0*G*(strainInc(i,5) - dep(5))
        if (nshr .ge. 3) stressNew(i,6) = stressOld(i,6) &
                       + 2.0d0*G*(strainInc(i,6) - dep(6))

        do j = 1, nstatev
            stateNew(i,j) = stateOld(i,j)
        end do
!       누적 소성변형률을 SDV 1-6 에 남긴다 (있으면)
        IF (nstatev .GE. 6) THEN
            do j = 1, 6
                stateNew(i,j) = stateOld(i,j) + dep(j)
            end do
        END IF

!       ---- FEMSTRESS 용으로 적분점 보관 -------------------------------
        IF (n_mp_ddm .LT. nmp_max) THEN
            n_mp_ddm = n_mp_ddm + 1
            mp_xyz(n_mp_ddm,1) = coordMp(i,1)
            mp_xyz(n_mp_ddm,2) = coordMp(i,2)
            mp_xyz(n_mp_ddm,3) = coordMp(i,3)
            do j = 1, 6
                mp_sig(n_mp_ddm,j) = 0.0d0
            end do
            mp_sig(n_mp_ddm,1) = stressNew(i,1)
            mp_sig(n_mp_ddm,2) = stressNew(i,2)
            mp_sig(n_mp_ddm,3) = stressNew(i,3)
            mp_sig(n_mp_ddm,4) = stressNew(i,4)
            if (nshr .ge. 2) mp_sig(n_mp_ddm,5) = stressNew(i,5)
            if (nshr .ge. 3) mp_sig(n_mp_ddm,6) = stressNew(i,6)
        END IF

    end do

    return
end subroutine vumat


subroutine vexternaldb(lOp, i_Array, niArray, r_Array, nrArray)
    USE IFPORT
    IMPLICIT NONE
    INTEGER            :: lOp, niArray, nrArray
    INTEGER            :: i_Array(niArray)
    DOUBLE PRECISION   :: r_Array(nrArray)
    LOGICAL            :: FILEEXISTANCE
    LOGICAL            :: STOP_EXISTS, SEGS_EXIST
    INTEGER            :: kInc
!  i_Array index map
    INTEGER, PARAMETER :: i_int_nTotalNodes    = 1
    INTEGER, PARAMETER :: i_int_nTotalElements = 2
    INTEGER, PARAMETER :: i_int_kStep          = 3
    INTEGER, PARAMETER :: i_int_kInc           = 4
    INTEGER, PARAMETER :: i_int_iStatus        = 5
    INTEGER, PARAMETER :: i_int_lWriteRestart  = 6
!  lOp codes
    INTEGER, PARAMETER :: j_int_StartAnalysis  = 0
    INTEGER, PARAMETER :: j_int_StartStep      = 1
    INTEGER, PARAMETER :: j_int_SetupIncrement = 2
    INTEGER, PARAMETER :: j_int_StartIncrement = 3
    INTEGER, PARAMETER :: j_int_EndIncrement   = 4
    INTEGER, PARAMETER :: j_int_EndStep        = 5
    INTEGER, PARAMETER :: j_int_EndAnalysis    = 6
!  r_Array indices
    INTEGER, PARAMETER :: i_flt_TotalTime=1, i_flt_StepTime=2, i_flt_dTime=3
!  flag
    CHARACTER(*), PARAMETER :: f_stress_ready_flag = "ABAQUS_stress_ready"
    CHARACTER(*), PARAMETER :: f_pause_flag = "ABAQUS_pause"
    CHARACTER(*), PARAMETER :: f_running_flag = "ABAQUS_running"
    CHARACTER(*), PARAMETER :: f_stop_flag = "ABAQUS_stop"
!   [CAUTION] foldername should be ABSOLUTE PATH !!!!
    CHARACTER(*), PARAMETER :: foldername = "/home/wnswjswk1/Codes/OpenDiS_new/tests/test21_eig_fem_cylinder/ABAQUS"
    CHARACTER(*), PARAMETER :: f_flag = ""
    CHARACTER(LEN=256) :: full_f_pause
    CHARACTER(LEN=256) :: full_f_stress_ready
    CHARACTER(LEN=256) :: full_f_running
    CHARACTER(LEN=256) :: full_f_stop
    CHARACTER(LEN=256) :: full_f_segs, full_f_femstress, full_f_surfstress

!   ---- DDM COMMON (vumat 과 같은 모양이어야 한다) ---------------------
    INTEGER, PARAMETER :: nseg_max = 20000
    DOUBLE PRECISION :: sg_old(nseg_max,6), sg_new(nseg_max,6)
    DOUBLE PRECISION :: sg_dep(nseg_max,6), sg_trv(nseg_max)
    DOUBLE PRECISION :: ddm_tstep, ddm_eff
    INTEGER :: sg_id(nseg_max,2), n_seg_ddm
    COMMON /ddmsegd/ sg_old, sg_new, sg_dep, sg_trv, ddm_tstep, ddm_eff
    COMMON /ddmsegi/ sg_id, n_seg_ddm

    INTEGER, PARAMETER :: nmp_max = 300000
    DOUBLE PRECISION :: mp_xyz(nmp_max,3), mp_sig(nmp_max,6)
    INTEGER :: n_mp_ddm
    COMMON /ddmmpd/ mp_xyz, mp_sig
    COMMON /ddmmpi/ n_mp_ddm

    DOUBLE PRECISION :: dyad(9), tsum, best, dd, pm(3)
    INTEGER :: k, j, nfin, ios, ibest

    full_f_pause        = TRIM(foldername)//"/"//TRIM(f_pause_flag)//TRIM(f_flag)//".flag"
    full_f_stress_ready = TRIM(foldername)//"/"//TRIM(f_stress_ready_flag)//TRIM(f_flag)//".flag"
    full_f_running      = TRIM(foldername)//"/"//TRIM(f_running_flag)//TRIM(f_flag)//".flag"
    full_f_stop         = TRIM(foldername)//"/"//TRIM(f_stop_flag)//TRIM(f_flag)//".flag"
    full_f_segs         = TRIM(foldername)//"/Data_segs.txt"
    full_f_femstress    = TRIM(foldername)//"/../FEMSTRESS"
    full_f_surfstress   = TRIM(foldername)//"/../SurfaceStress"

    SELECT CASE (lOp)

    CASE (j_int_StartAnalysis)
        n_seg_ddm = 0
        n_mp_ddm  = 0
        ddm_tstep = 0.0d0
        ddm_eff   = 0.0d0

    CASE (j_int_StartIncrement)
        INQUIRE(FILE=full_f_stop, EXIST=STOP_EXISTS)
        IF (STOP_EXISTS) THEN
            PRINT *, "VEXTERNALDB: ABAQUS_stop.flag detected at StartIncrement. Terminating..."
            CALL XPLB_EXIT
        END IF

!       이 증분의 적분점 보관함을 비운다
        n_mp_ddm = 0

!       OpenDiS 가 쓴 세그먼트를 읽는다 (헤더 + 23 열)
!         n_segment n_segment_final t_step EffSlip
!         id1 id2 | r1_old(3) r2_old(3) | r1_new(3) r2_new(3) | dyad(9)
        INQUIRE(FILE=full_f_segs, EXIST=SEGS_EXIST)
        IF (SEGS_EXIST) THEN
            OPEN(UNIT=81, FILE=TRIM(full_f_segs), STATUS="OLD", ACTION="READ", IOSTAT=ios)
            IF (ios .EQ. 0) THEN
                READ(81,*,IOSTAT=ios) n_seg_ddm, nfin, ddm_tstep, ddm_eff
                IF (ios .NE. 0) n_seg_ddm = 0
                IF (n_seg_ddm .GT. nseg_max) n_seg_ddm = nseg_max
                DO k = 1, n_seg_ddm
                    READ(81,*,IOSTAT=ios) sg_id(k,1), sg_id(k,2), &
                        (sg_old(k,j), j=1,6), (sg_new(k,j), j=1,6), (dyad(j), j=1,9)
                    IF (ios .NE. 0) THEN
                        n_seg_ddm = k - 1
                        EXIT
                    END IF
!                   세그먼트 중점이 움직인 거리 (원본 :1134-1137)
                    tsum = 0.0d0
                    DO j = 1, 3
                        tsum = tsum + (sg_new(k,j) + sg_new(k,j+3) &
                                     - sg_old(k,j) - sg_old(k,j+3))**2
                    END DO
                    sg_trv(k) = 0.5d0*sqrt(tsum)
!                   dyad = outer(n_hat, b) -> 대칭부가 de_p (원본 :1842-1847)
                    sg_dep(k,1) = dyad(1)
                    sg_dep(k,2) = dyad(5)
                    sg_dep(k,3) = dyad(9)
                    sg_dep(k,4) = 0.5d0*(dyad(2) + dyad(4))
                    sg_dep(k,5) = 0.5d0*(dyad(6) + dyad(8))
                    sg_dep(k,6) = 0.5d0*(dyad(7) + dyad(3))
                END DO
                CLOSE(81)
                PRINT *, "VEXTERNALDB: read Data_segs.txt, n_seg =", n_seg_ddm, &
                         " t_step =", ddm_tstep, " EffSlip =", ddm_eff
            END IF
        END IF

    CASE (j_int_EndIncrement)
        PRINT *, "EndInc: kInc=", i_Array(i_int_kInc), &
           " StepTime=", r_Array(i_flt_StepTime), " dt=", r_Array(i_flt_dTime)
        INQUIRE(FILE=full_f_stop, EXIST=STOP_EXISTS)
        IF (STOP_EXISTS) THEN
            PRINT *, "VEXTERNALDB: ABAQUS_stop.flag detected at EndIncrement. Terminating..."
            CALL XPLB_EXIT
        END IF

!       ---- FEMSTRESS: 세그먼트마다 가장 가까운 적분점의 응력 ----------
!       OpenDiS 의 ReadRemoteStress 가 읽는 형식:
!         n1 n2 sxx syy szz syz sxz sxy flag        (9 항)
        IF (n_seg_ddm .GT. 0 .AND. n_mp_ddm .GT. 0) THEN
            OPEN(UNIT=82, FILE=TRIM(full_f_femstress), STATUS="REPLACE", ACTION="WRITE")
            DO k = 1, n_seg_ddm
                DO j = 1, 3
                    pm(j) = 0.25d0*(sg_old(k,j) + sg_old(k,j+3) &
                                  + sg_new(k,j) + sg_new(k,j+3))
                END DO
                ibest = 1
                best  = 1.0d30
                DO j = 1, n_mp_ddm
                    dd = (mp_xyz(j,1)-pm(1))**2 + (mp_xyz(j,2)-pm(2))**2 &
                       + (mp_xyz(j,3)-pm(3))**2
                    IF (dd .LT. best) THEN
                        best  = dd
                        ibest = j
                    END IF
                END DO
!               ABAQUS VUMAT 의 응력 순서는 (11,22,33,12,23,31) 인데
!               FEMSTRESS 는 (xx,yy,zz,yz,xz,xy) 로 읽힌다 -> 전단을 바꿔 쓴다
                WRITE(82,'(I8,1X,I8,6(1X,E16.8),1X,I2)') &
                    sg_id(k,1), sg_id(k,2), &
                    mp_sig(ibest,1), mp_sig(ibest,2), mp_sig(ibest,3), &
                    mp_sig(ibest,5), mp_sig(ibest,6), mp_sig(ibest,4), 1
            END DO
            CLOSE(82)
            PRINT *, "VEXTERNALDB: wrote FEMSTRESS for", n_seg_ddm, "segments from", &
                     n_mp_ddm, "integration points"
        END IF

!       ---- SurfaceStress: 핵생성이 보는 자리별 응력 --------------------
!       make_nuc_sites 가 읽는 10 열 형식:
!         id  x  y  z  S11 S22 S33 S23 S13 S12
!       생산 DDM 에서 Set-SurfaceEl 이 사실상 전체 요소이므로 (STATUS 문서 참고)
!       여기서도 적분점을 전부 자리로 쓴다.
!       (11/17/2026, junho) 적분점마다 실제 응력을 쓴다.
        IF (n_mp_ddm .GT. 0) THEN
            OPEN(UNIT=83, FILE=TRIM(full_f_surfstress), STATUS="REPLACE", ACTION="WRITE")
            DO j = 1, n_mp_ddm
                WRITE(83,'(I8,9(1X,E16.8))') &
                    j, mp_xyz(j,1), mp_xyz(j,2), mp_xyz(j,3), &
                    mp_sig(j,1), mp_sig(j,2), mp_sig(j,3), &
                    mp_sig(j,5), mp_sig(j,6), mp_sig(j,4)
            END DO
            CLOSE(83)
            PRINT *, "VEXTERNALDB: wrote SurfaceStress with", n_mp_ddm, "sites"
        END IF

        kInc = i_Array(i_int_kInc)
        IF (kInc == 1) THEN
            PRINT *, "[EndStep] kInc = 1. write ABAQUS_PAUSE.flag only"
            OPEN(UNIT=97, FILE=TRIM(full_f_running), STATUS="REPLACE", ACTION="WRITE")
            CLOSE(97, STATUS="DELETE")
            OPEN(UNIT=99, FILE=TRIM(full_f_pause), STATUS="REPLACE", ACTION="WRITE")
            CLOSE(99)
            DO WHILE (.TRUE.)
                INQUIRE (FILE=full_f_pause, EXIST=FILEEXISTANCE)
                IF(FILEEXISTANCE) THEN
                    CALL SLEEPQQ(1000)
                ELSE
                    EXIT
                END IF
            END DO
        ELSE
            PRINT *, "[EndStep] kInc = ", kInc, ". write ABAQUS_PAUSE.flag and ABAQUS_stress_ready.flag"
            OPEN(UNIT=97, FILE=TRIM(full_f_running), STATUS="REPLACE", ACTION="WRITE")
            CLOSE(97, STATUS="DELETE")
            OPEN(UNIT=99, FILE=TRIM(full_f_pause), STATUS="REPLACE", ACTION="WRITE")
            CLOSE(99)
            OPEN(UNIT=99, FILE=TRIM(full_f_stress_ready), STATUS="REPLACE", ACTION="WRITE")
            CLOSE(99)
            DO WHILE (.TRUE.)
                INQUIRE (FILE=full_f_pause, EXIST=FILEEXISTANCE)
                IF(FILEEXISTANCE) THEN
                    CALL SLEEPQQ(1000)
                ELSE
                    EXIT
                END IF
            END DO
        ENDIF

    CASE (j_int_EndAnalysis)
        PRINT *, "[EndAnalysis] Clean up flags ..."
        INQUIRE(FILE=TRIM(full_f_pause), EXIST=FILEEXISTANCE)
        IF (FILEEXISTANCE) THEN
            OPEN(99, FILE=TRIM(full_f_pause), STATUS="OLD")
            CLOSE(99, STATUS="DELETE")
        END IF
        INQUIRE(FILE=TRIM(full_f_stress_ready), EXIST=FILEEXISTANCE)
        IF (FILEEXISTANCE) THEN
            OPEN(98, FILE=TRIM(full_f_stress_ready), STATUS="OLD")
            CLOSE(98, STATUS="DELETE")
        END IF
        INQUIRE(FILE=TRIM(full_f_running), EXIST=FILEEXISTANCE)
        IF (FILEEXISTANCE) THEN
            OPEN(97, FILE=TRIM(full_f_running), STATUS="OLD")
            CLOSE(97, STATUS="DELETE")
        END IF
        PRINT *, "TotalTime=", r_Array(i_flt_TotalTime), &
                 ", LastStepTime=", r_Array(i_flt_StepTime)
        CALL FLUSH(6)
    END SELECT
    RETURN
END subroutine vexternaldb
