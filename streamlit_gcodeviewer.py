import streamlit as st
import numpy as np
import re
import plotly.graph_objects as go
import tempfile
import chardet
from typing import Tuple, List

st.set_page_config(layout="wide")

# ------------------------------------------------------------
# G-code 파싱
# ------------------------------------------------------------
def parse_gcode(file_path: str) -> Tuple[np.ndarray, List[bool], float]:
    coords_buffer = []
    is_extrudes_buffer = []
    f_value = 0.0
    last_pos = {'X': None, 'Y': None, 'Z': None}
    current_extrude = False

    with open(file_path, 'rb') as raw_file:
        raw_data = raw_file.read()
        enc = (chardet.detect(raw_data).get('encoding') or 'utf-8').strip() or 'utf-8'

    lines = raw_data.decode(enc, errors='replace').splitlines()
    kv_re  = re.compile(r'([XYZEFe])([-+]?[0-9]*\.?[0-9]+)')

    for line in lines:
        s = line.strip()
        if not s or s.startswith(';'):
            continue

        found_any = False
        matches = kv_re.findall(s)
        for axis, val in matches:
            a = axis.upper()
            if a in last_pos:
                last_pos[a] = float(val)
                found_any = True
            elif a == 'F':
                f_value = float(val)
            elif a == 'E':
                current_extrude = float(val) > 0

        if found_any and None not in last_pos.values():
            coords_buffer.append([last_pos['X'], last_pos['Y'], last_pos['Z']])
            is_extrudes_buffer.append(current_extrude)

    coords = np.array(coords_buffer, dtype=float) if coords_buffer else np.empty((0,3), float)
    # 방어: 길이 보정
    if len(is_extrudes_buffer) != len(coords):
        if len(coords) > 0:
            last = is_extrudes_buffer[-1] if is_extrudes_buffer else False
            is_extrudes_buffer = (is_extrudes_buffer + [last] * (len(coords) - len(is_extrudes_buffer)))[:len(coords)]
        else:
            is_extrudes_buffer = []

    return coords, is_extrudes_buffer, f_value

# ------------------------------------------------------------
# 경로 누적선(선분) 만들기
#  - is_extrudes[i] 상태(도착점 기준)가 True/False에 따라
#    각 선분(i-1 -> i)을 두 묶음(압출/이동)으로 분리해 NaN으로 구획
#  - Z 컷오프 적용: 두 점 모두 Z > limit이면 그 선분은 제외
# ------------------------------------------------------------
def build_series(coords: np.ndarray, is_extrudes: List[bool], z_limit: float, upto_idx: int = None):
    n = len(coords)
    if n < 2:
        return (np.array([]), np.array([]), np.array([]),
                np.array([]), np.array([]), np.array([]))

    if upto_idx is None:
        upto_idx = n - 1
    upto_idx = int(max(1, min(upto_idx, n - 1)))

    ex_x, ex_y, ex_z = [], [], []
    mv_x, mv_y, mv_z = [], [], []

    def push(seglist, p0, p1):
        # 선분을 (p0, p1, NaN) 형태로 적재 → 분절 유지
        seglist[0].extend([p0[0], p1[0], np.nan])
        seglist[1].extend([p0[1], p1[1], np.nan])
        seglist[2].extend([p0[2], p1[2], np.nan])

    for i in range(1, upto_idx + 1):
        p0 = coords[i - 1]
        p1 = coords[i]
        # Z 컷: 두 점 모두 컷오프보다 위면 스킵
        if p0[2] > z_limit and p1[2] > z_limit:
            continue
        if bool(is_extrudes[i]):
            push((ex_x, ex_y, ex_z), p0, p1)
        else:
            push((mv_x, mv_y, mv_z), p0, p1)

    return np.array(ex_x), np.array(ex_y), np.array(ex_z), np.array(mv_x), np.array(mv_y), np.array(mv_z)

# ------------------------------------------------------------
# Plotly 애니메이션 Figure 생성
#  - frames_count를 제한하여 최대 프레임 수를 cap (기본 800)
#  - 프레임마다 upto_idx가 증가 (step 크기)
#  - 클라이언트 측에서 재생/일시정지/속도 버튼으로 제어
# ------------------------------------------------------------
def build_animated_figure(coords: np.ndarray,
                          is_extrudes: List[bool],
                          z_limit: float,
                          max_frames: int = 800,
                          caption: str = "") -> go.Figure:
    n = len(coords)
    total_lines = max(0, n - 1)
    if total_lines == 0:
        return go.Figure()

    # 프레임 수 제한
    step = max(1, total_lines // max_frames)
    last_idx = (total_lines // step) * step
    if last_idx < total_lines:
        last_idx = total_lines

    # 초기 데이터 (idx = step)
    ex_x, ex_y, ex_z, mv_x, mv_y, mv_z = build_series(coords, is_extrudes, z_limit, upto_idx=step)

    fig = go.Figure(
        data=[
            go.Scatter3d(x=ex_x, y=ex_y, z=ex_z, mode='lines',
                         line=dict(color='blue', width=2), name='Extrude(E>0)'),
            go.Scatter3d(x=mv_x, y=mv_y, z=mv_z, mode='lines',
                         line=dict(color='gray', width=1), name='Travel(E=0)')
        ],
        layout=go.Layout(
            scene=dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            height=700,
            updatemenus=[
                # 재생/일시정지
                dict(
                    type="buttons",
                    direction="left",
                    x=0.0, y=1.12, xanchor="left", yanchor="top",
                    buttons=[
                        dict(label="▶ Play (1×)", method="animate",
                             args=[None, {"frame": {"duration": 30, "redraw": False},
                                          "fromcurrent": True,
                                          "transition": {"duration": 0}}]),
                        dict(label="⏸ Pause", method="animate",
                             args=[[None], {"frame": {"duration": 0, "redraw": False},
                                            "mode": "immediate",
                                            "transition": {"duration": 0}}]),
                    ],
                ),
                # 속도 프리셋
                dict(
                    type="buttons",
                    direction="left",
                    x=0.0, y=1.06, xanchor="left", yanchor="top",
                    buttons=[
                        dict(label="0.25×", method="animate",
                             args=[None, {"frame": {"duration": 120, "redraw": False},
                                          "fromcurrent": True, "transition": {"duration": 0}}]),
                        dict(label="1×", method="animate",
                             args=[None, {"frame": {"duration": 30, "redraw": False},
                                          "fromcurrent": True, "transition": {"duration": 0}}]),
                        dict(label="2×", method="animate",
                             args=[None, {"frame": {"duration": 15, "redraw": False},
                                          "fromcurrent": True, "transition": {"duration": 0}}]),
                        dict(label="4×", method="animate",
                             args=[None, {"frame": {"duration": 7, "redraw": False},
                                          "fromcurrent": True, "transition": {"duration": 0}}]),
                    ],
                ),
            ],
            sliders=[dict(
                active=0,
                y=1.0, x=0.0, len=1.0, xanchor="left",
                steps=[
                    dict(
                        method="animate",
                        label=str(k),
                        args=[
                            [f"frame_{k}"],
                            {"mode": "immediate", "frame": {"duration": 0, "redraw": False},
                             "transition": {"duration": 0}}
                        ],
                    )
                    for k in range(step, last_idx + 1, step)
                ],
            )],
            annotations=[
                dict(text=caption, x=0, y=1.16, xref="paper", yref="paper",
                     xanchor="left", yanchor="top", showarrow=False)
            ]
        ),
        frames=[]
    )

    # 프레임 생성 (서버에서 1회 준비 → 클라이언트가 재생)
    frames = []
    for k in range(step, last_idx + 1, step):
        ex_x, ex_y, ex_z, mv_x, mv_y, mv_z = build_series(coords, is_extrudes, z_limit, upto_idx=k)
        frames.append(go.Frame(
            name=f"frame_{k}",
            data=[
                go.Scatter3d(x=ex_x, y=ex_y, z=ex_z),
                go.Scatter3d(x=mv_x, y=mv_y, z=mv_z)
            ]
        ))
    fig.frames = frames
    return fig

# ------------------------------------------------------------
# 거리 계산
# ------------------------------------------------------------
def compute_total_distance(coords: np.ndarray) -> float:
    if len(coords) < 2:
        return 0.0
    diffs = np.diff(coords, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("🧠 G-code 3D Viewer — 클라이언트 측 애니메이션(무깜빡임)")

uploaded_file = st.file_uploader("G-code 파일 업로드", type=["gcode", "nc"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".gcode") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    coords, is_extrudes, f_value = parse_gcode(temp_path)

    if len(coords) < 2:
        st.error("⚠ G-code 내 유효한 좌표가 부족합니다.")
        st.stop()

    total_distance = compute_total_distance(coords)
    z_min, z_max = float(coords[:, 2].min()), float(coords[:, 2].max())

    col_stats, col_opts = st.columns([2, 1])
    with col_stats:
        st.markdown(f"""
- 총 세그먼트 수: **{len(coords)-1}**
- 총 이동 거리: **{total_distance:.2f} mm**
- F값 (이송 속도): **{f_value:.1f} mm/min**
- 예상 소요 시간: **{(total_distance / f_value) if f_value>0 else 0.0:.2f} 분**
- Z 범위: **{z_min:.1f} ~ {z_max:.1f} mm**
""")

    with col_opts:
        z_limit = st.slider("🧱 Z 진행 높이 (mm)", min_value=z_min, max_value=z_max, value=z_max, step=1.0)
        max_frames = st.number_input("🎞 최대 프레임 수 cap", min_value=100, max_value=5000, value=800, step=100,
                                     help="프레임이 너무 많으면 무거워집니다. cap으로 부드럽게.")
        st.caption("• Play/Pause와 속도(0.25×/1×/2×/4×)는 그래프 내부 버튼으로 제어")

    # 캡션(좌상단 안내)
    caption = f"Frames ~ {min(len(coords)-1, max_frames)} (step 자동 조절) | Z ≤ {z_limit:.1f}mm"

    fig = build_animated_figure(coords, is_extrudes, z_limit, max_frames=max_frames, caption=caption)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
st.markdown("""
**📘 사용 방법**
1. `.gcode` 또는 `.nc` 파일 업로드
2. **Z 진행 높이**를 설정하고, 필요시 **최대 프레임 수 cap**을 조절
3. 그래프 좌상단의 **Play / Pause** 및 **속도 버튼(0.25×/1×/2×/4×)**으로 부드러운 재생
4. 슬라이더로 특정 프레임(라인 진행 위치)을 수동 탐색 가능
""")
