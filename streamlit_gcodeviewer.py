import streamlit as st
import numpy as np
import re
import plotly.graph_objects as go
import tempfile
import time
import chardet

st.set_page_config(layout="wide")

# ------------------------------------------------------------
#  상태 초기화
# ------------------------------------------------------------
def _init_state():
    if "idx" not in st.session_state:
        st.session_state.idx = 1          # 현재 그려줄 마지막 좌표 인덱스 (선분 기준)
    if "playing" not in st.session_state:
        st.session_state.playing = False   # 재생 중 여부
    if "speed_lps" not in st.session_state:
        st.session_state.speed_lps = 50    # 초당 라인 수 (lines per second)
    if "z_limit" not in st.session_state:
        st.session_state.z_limit = None    # 슬라이더 값 유지용
    if "coords" not in st.session_state:
        st.session_state.coords = None
    if "is_extrudes" not in st.session_state:
        st.session_state.is_extrudes = None
    if "f_value" not in st.session_state:
        st.session_state.f_value = 0.0
_init_state()

# ------------------------------------------------------------
#  G-code 파싱 함수
# ------------------------------------------------------------
def parse_gcode(file_path):
    coords_buffer = []
    is_extrudes_buffer = []
    f_value = 0.0
    last_pos = {'X': None, 'Y': None, 'Z': None}
    current_extrude = False

    with open(file_path, 'rb') as raw_file:
        raw_data = raw_file.read()
        detected = chardet.detect(raw_data)
        encoding = detected.get('encoding') or 'utf-8'

    lines = raw_data.decode(encoding, errors='replace').splitlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith(";"):
            continue

        # 좌표/피드/압출 추출
        matches = re.findall(r'([XYZEFe])([-+]?[0-9]*\.?[0-9]+)', line)
        found_e = False
        for axis, value in matches:
            axis_upper = axis.upper()
            if axis_upper in last_pos:
                last_pos[axis_upper] = float(value)
            elif axis_upper == 'F':
                f_value = float(value)
            elif axis_upper == 'E':
                current_extrude = float(value) > 0
                found_e = True

        # X,Y,Z 모두 있으면 점 추가
        if None not in last_pos.values():
            coords_buffer.append([last_pos['X'], last_pos['Y'], last_pos['Z']])
            # 해당 라인에 E가 없으면 직전 상태 유지
            is_extrudes_buffer.append(current_extrude)

    coords = np.array(coords_buffer) if coords_buffer else np.empty((0, 3), dtype=float)
    return coords, is_extrudes_buffer, f_value

# ------------------------------------------------------------
#  거리 계산 함수
# ------------------------------------------------------------
def compute_total_distance(coords):
    if len(coords) < 2:
        return 0.0
    diffs = np.diff(coords, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return float(np.sum(distances))

# ------------------------------------------------------------
#  분기/누적 인덱스 기반 시각화 (E값 색/스타일 + Z제한 + 인덱스 제한)
# ------------------------------------------------------------
def plot_path_partial(coords, is_extrudes, max_index, max_z):
    # max_index: 마지막으로 포함할 좌표 인덱스 (최소 1)
    max_index = int(max(1, min(max_index, len(coords) - 1)))

    def group_segments(target_extrude: bool):
        group = []
        grouped = []
        # 선분은 (i-1 -> i) 이므로 i=1..max_index 까지
        for i in range(1, max_index + 1):
            z0 = coords[i - 1][2]
            z1 = coords[i][2]
            # 둘 다 z 제한보다 크면 스킵 (보이지 않게)
            if z0 > max_z and z1 > max_z:
                if group:
                    grouped.append(np.array(group))
                    group = []
                continue

            if bool(is_extrudes[i]) == target_extrude:
                if not group:
                    group.append(coords[i - 1])
                group.append(coords[i])
            else:
                if group:
                    grouped.append(np.array(group))
                    group = []
        if group:
            grouped.append(np.array(group))
        return grouped

    fig = go.Figure()

    # 🔵 파란 실선 (E>0)
    ex_groups = group_segments(True)
    for seg in ex_groups:
        a = seg.T
        fig.add_trace(go.Scatter3d(
            x=a[0], y=a[1], z=a[2],
            mode='lines',
            line=dict(color='blue', width=2),
            showlegend=False
        ))

    # ⚪ 회색 실선 (이동만)
    mv_groups = group_segments(False)
    for seg in mv_groups:
        a = seg.T
        fig.add_trace(go.Scatter3d(
            x=a[0], y=a[1], z=a[2],
            mode='lines',
            line=dict(color='gray', width=1),
            showlegend=False
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=700
    )
    return fig

# ------------------------------------------------------------
#  재생 제어 콜백
# ------------------------------------------------------------
def on_play():
    st.session_state.playing = True

def on_pause():
    st.session_state.playing = False

def on_stop():
    st.session_state.playing = False
    st.session_state.idx = 1

def step_forward():
    if st.session_state.coords is None:
        return
    st.session_state.idx = min(st.session_state.idx + 1, len(st.session_state.coords) - 1)

def step_backward():
    st.session_state.idx = max(1, st.session_state.idx - 1)

# ------------------------------------------------------------
#  UI
# ------------------------------------------------------------
st.title("🧠 G-code 3D Viewer (플레이/정지/속도조절 지원, E값: 파란/회색 실선)")
uploaded_file = st.file_uploader("G-code 파일 업로드", type=["gcode", "nc"])

# 파일 업로드 및 파싱
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".gcode") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    coords, is_extrudes, f_value = parse_gcode(temp_path)

    if len(coords) < 2:
        st.error("⚠ G-code 내 유효한 좌표가 부족합니다.")
    else:
        # 상태에 로드
        st.session_state.coords = coords
        st.session_state.is_extrudes = is_extrudes
        st.session_state.f_value = f_value

        # 기본 인덱스 보정
        st.session_state.idx = min(max(1, st.session_state.idx), len(coords) - 1)

        total_distance = compute_total_distance(coords)
        z_min, z_max = float(coords[:, 2].min()), float(coords[:, 2].max())
        if st.session_state.z_limit is None:
            st.session_state.z_limit = z_max

        est_time_min = (total_distance / f_value) if f_value > 0 else 0.0
        # f 값이 mm/min 가정 → 총 거리(mm)/mm/min = 분

        col_stats, col_ctrl = st.columns([2, 1])
        with col_stats:
            st.markdown(f"""
- 총 세그먼트 수: **{len(coords)-1}**
- 총 이동 거리: **{total_distance:.2f} mm**
- F값 (이송 속도): **{f_value:.1f} mm/min**
- 예상 소요 시간: **{est_time_min:.2f} 분**
- Z 범위: **{z_min:.1f} ~ {z_max:.1f} mm**
- 현재 표시 라인(선분) 인덱스: **{st.session_state.idx}/{len(coords)-1}**
""")

        with col_ctrl:
            st.session_state.z_limit = st.slider(
                "🧱 Z 진행 높이 (mm)",
                min_value=z_min, max_value=z_max,
                value=float(st.session_state.z_limit),
                step=1.0
            )
            st.session_state.speed_lps = st.slider(
                "⏩ 재생 속도 (초당 라인 수)",
                min_value=1, max_value=200, value=int(st.session_state.speed_lps), step=1
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.button("▶️ Play", use_container_width=True, on_click=on_play)
            with c2:
                st.button("⏸️ Pause", use_container_width=True, on_click=on_pause)
            with c3:
                st.button("⏹️ Stop", use_container_width=True, on_click=on_stop)

            c4, c5 = st.columns(2)
            with c4:
                st.button("⬅️ Step -1", use_container_width=True, on_click=step_backward)
            with c5:
                st.button("Step +1 ➡️", use_container_width=True, on_click=step_forward)

        # 그래프
        with st.expander("🔍 경로 시각화 보기", expanded=True):
            fig = plot_path_partial(coords, is_extrudes, st.session_state.idx, st.session_state.z_limit)
            st.plotly_chart(fig, use_container_width=True)

        # 재생 루프 (Play 중일 때만 한 프레임 전진 후 재실행)
        if st.session_state.playing:
            # 인덱스 증가
            if st.session_state.idx < len(coords) - 1:
                st.session_state.idx = min(
                    st.session_state.idx + max(1, int(st.session_state.speed_lps)),
                    len(coords) - 1
                )
            else:
                # 끝에 도달하면 자동으로 정지
                st.session_state.playing = False

            # 속도에 따른 sleep 후 재실행
            # speed_lps: 초당 라인 수 → 프레임당 1회 갱신이라면 대략 1/s 초 슬립
            # (한 번에 여러 라인을 넘어가므로 아래 슬립은 부드러운 재생 타이밍 용도)
            delay = 1.0 / 30.0  # 대략 30FPS 템포로 UI 갱신
            time.sleep(delay)
            st.experimental_rerun()

# ------------------------------------------------------------
#  사용 설명
# ------------------------------------------------------------
st.markdown("""
**📘 사용 방법**
1. `.gcode` 또는 `.nc` 형식의 G-code 파일을 업로드하세요.
2. E값이 있는 압출 구간은 **파란색 실선**, 없는 이동 구간은 **얇은 회색 실선**으로 구분됩니다.
3. Z 슬라이더로 출력 높이를 제한하며 경로를 확인할 수 있습니다.
4. **Play / Pause / Stop** 버튼으로 애니메이션 재생을 제어하고, **초당 라인 수** 슬라이더로 속도를 조절하세요.
5. **Step -1 / Step +1** 버튼으로 1라인(선분) 단위로 수동 이동이 가능합니다.
""")
