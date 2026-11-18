# -*- coding: utf-8 -*-
u"""생산 VUMAT 이 읽는 메쉬 기반 입력 두 개를 .inp 에서 만든다.

형식은 ParaDiS MATLAB 쪽 생산자를 그대로 따른다:
  surface_elements.txt    DDM_Linux_v0.m:48    '%d  %d\n'  (elem_id, flag)
                          import_elements_nodes.m:80  표면이면 flag = 6
  OriginalCoordNodes.txt  import_elements_nodes.m:96   '%d' + 24 x '%.8e'
                          (C3D8 = Elements_col_num 9)

표면 요소 판정: 정육면체 바깥면에 절점이 하나라도 닿으면 표면.
"""
import os
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

INP = sys.argv[1] if len(sys.argv) > 1 else 'test_eig_fem_elastic_VUMAT_control.inp'
OUT = os.path.dirname(os.path.abspath(INP))
TOL = 1e-6


def parse(path):
    nodes, elems, mode = {}, [], None
    for raw in open(path):
        t = raw.strip()
        if not t:
            continue
        if t.startswith('*'):
            low = t.lower()
            if low.startswith('*node'):
                mode = 'n'
            elif low.startswith('*element'):
                mode = 'e'
            else:
                mode = None
            continue
        if mode is None:
            continue
        p = [x.strip() for x in t.split(',')]
        try:
            if mode == 'n' and len(p) >= 4:
                nodes[int(p[0])] = (float(p[1]), float(p[2]), float(p[3]))
            elif mode == 'e' and len(p) >= 9:
                elems.append((int(p[0]), [int(x) for x in p[1:9]]))
        except ValueError:
            continue
    return nodes, elems


nodes, elems = parse(INP)
print(u'절점 %d   요소 %d' % (len(nodes), len(elems)))
if not nodes or not elems:
    sys.exit(u'파싱 실패')

# A face that belongs to exactly one element is on the surface of the
# body.  That is true whatever the shape is, so a cylinder works as well
# as a box.  MATLAB then calls an element a surface element when it has
# ANY node on the surface (import_elements_nodes.m:80), which is what the
# node set below reproduces -- not "has a free face", which is slightly
# narrower.
#   (11/18/2026, junho) was a bounding-box test, which only works on a box.
FACES = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
         (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]

face_count = {}
for eid, conn in elems:
    for f in FACES:
        key = tuple(sorted(conn[i] for i in f))
        face_count[key] = face_count.get(key, 0) + 1

boundary_nodes = set()
nfree = 0
for key, c in face_count.items():
    if c == 1:
        nfree += 1
        boundary_nodes.update(key)
print(u'자유면 %d, 표면 절점 %d / 전체 %d'
      % (nfree, len(boundary_nodes), len(nodes)))

xs = [c[0] for c in nodes.values()]
ys = [c[1] for c in nodes.values()]
zs = [c[2] for c in nodes.values()]
print(u'범위  x %.1f~%.1f  y %.1f~%.1f  z %.1f~%.1f'
      % (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)))


def on_boundary_node(n):
    return n in boundary_nodes


# ------------------------------------------------ surface_elements.txt
p1 = os.path.join(OUT, 'surface_elements.txt')
nsurf = 0
with open(p1, 'w') as f:
    for eid, conn in elems:
        flag = 6 if any(on_boundary_node(n) for n in conn) else 0
        if flag:
            nsurf += 1
        f.write('%d  %d\n' % (eid, flag))
print(u'%s  -> 표면 %d / 전체 %d  (내부 %d)'
      % (os.path.basename(p1), nsurf, len(elems), len(elems) - nsurf))

# ------------------------------------------------ OriginalCoordNodes.txt
p2 = os.path.join(OUT, 'OriginalCoordNodes.txt')
with open(p2, 'w') as f:
    for eid, conn in elems:
        vals = []
        for n in conn:
            vals.extend(nodes[n])
        f.write('%d ' % eid + ' '.join('%.8e' % v for v in vals) + '\n')
print(u'%s  -> %d 줄, 줄당 1+24 필드' % (os.path.basename(p2), len(elems)))

# ------------------------------------------------ 검산
with open(p2) as f:
    first = f.readline().split()
print(u'   첫 줄 필드 수 %d (25 여야 함)' % len(first))
eid0, conn0 = elems[0]
print(u'   요소 %d 의 1번 절점 %s  vs  파일 %s'
      % (eid0, nodes[conn0[0]], tuple(float(x) for x in first[1:4])))
