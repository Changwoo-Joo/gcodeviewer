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
    ss = st.session_state
    ss.setdefault("idx", 1)              # 현재 표시 마지막 선분 인덱스 (1..N-1)
    ss.setdefault("playing", False)      # 재생 중 여부
    ss.setdefault("speed_lps", 50)       # 초당 라인 수 (lines per second)
    ss.setdefault("z_limit", None)       # Z 컷오프
    ss.setdefault("coords", None)
    ss.setdefault("is_extrudes", None)
    ss.setdefault("f_value", 0.0)
    ss.setdefault("last_tick", 0.0)      # 재생 타이밍 제어
    ss.setdefault("target_fps", 30.0)    # UI 갱신 템포 (고정)
    ss.setdefault("loop_play", False)    # 끝에서 루프 재생 옵션

_init_state()

# ------------------------------------------------------------
#  G-code 파싱
# ------------------------------------------------------------
def parse_gcode(file_path):
    coords_buffer = []
    is_extrudes_buffer = []
    f_value = 0.0
    last_pos = {'X': None, 'Y': None, 'Z': None}
    current_extrude = False

    with open(file_path, 'rb') as raw_file:
        raw_data = raw_file.read()
        enc = (chardet.detect(raw_data).get('encoding') or 'utf-8').strip() or 'utf-8'

    lines = raw_data.decode(enc, errors='replace').splitlines()

    # G0/G1 라인만 대충 필터 (주석/공백은 스킵)
    cmd_re = re.compile(r'^[GMT]\d+', re.IGNORECASE)
    kv_re  = re.compile(r'([XYZEFe])([-+]?[0-9]*\.?[0-9]+)')

    for line in lines:
        s = line.strip()
        if not s or s.startswith(';'):
            continue
        # 좌표/피드/압출 추출
        matches = kv_re.findall(s)
        found_any = False
        for axis, val in matches:
            a = axis.upper()
            if a in last_pos:
                last_pos[a] = float(val); found_any = True
            elif a == 'F':
                f_value = float(val)
            elif a == 'E':
                current_extrude = float(val) > 0
        # X,Y,Z 모두 있으면 점 기록
        if found_any and None not in last_pos.values():
            coords_buffer.append([last_pos['X'], last_pos['Y'], last_pos['Z']])
            is_extrudes_buffer.append(current_extrude)

    coords = np.array(coords_buffer, dtype=float) if coords_buffer else np.empty((0,3), float)
    # 방어: 길이 보정
    if len(is_extrudes_buffer) != len(coords):
        if len(coords) > 0:
            # 부족하면 마지막 상태로 패딩
            last = is_extrudes_buffer[-1] if is_extrudes_buffer else False
            is_extrudes_buffer = (is_extrudes_buffer + [last] * (len(coords) - len(is_extrudes_buffer)))[:len(coords)]
        else:
            is_extrudes_buffer = []
    return coords, is_extrudes_buffer, f_value

# ------------------------------------------------------------
#  거리 계산
# ------------------------------------------------------------
def compute_total_distance(coords):
    if len(coords) < 2:
        return 0.0
    diffs = np.diff(coords, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())

# ------------------------------------------------------------
#  부분 시각화 (E값 색/스타일 + Z제한 + 인덱스 제한)
# ------------------------------------------------------------
def plot_path_partial(coords, is_extrudes, max_index, max_z):
    if len(coords) < 2:
        return go.Figure()
    max_index = int(max(1, min(max_index, len(coords) - 1)))

    def group_segments(target_extrude: bool):
        group = []
        grouped = []
        for i in range(1, max_index + 1):
            z0 = coords[i - 1][2]; z1 = coords[i][2]
            # 둘 다 제한 상단이면 숨김
            if z0 > max_z and z1 > max_z:
                if group:
                    grouped.append(np.array(group)); group = []
                continue
            # is_extrudes는 포인트 상태 → 도착점 기준
            if bool(is_extrudes[i]) == target_extrude:
                if not group:
                    group.append(coords[i - 1])
                group.append(coords[i])
            else:
                if group:
                    grouped.append(np.array(group)); group = []
        if group:
            grouped.append(np.array(group))
        return grouped

    fig = go.Figure()

    # 🔵 압출 구간
    for seg in group_segments(True):
        a = seg.T
        fig.add_trace(go.Scatter3d(
            x=a[0], y=a[1], z=a[2],
            mode='lines',
            line=dict(color='blue', width=2),
            showlegend=False
        ))
    # ⚪ 이동 구간
    for seg in group_segments(False):
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
    st.session_state.last_tick = time.time()

def on_pause():
    st.session_state.playing = False

def on_stop():
    st.session_state.playing = False
    st.session_state.idx = 1

def step_forward():
    coords = st.session_state.coords
    if coords is None: return
    st.session_state.idx = min(st.session_state.idx + 1, len(coords) - 1)

def step_backward():
    st.session_state.idx = max(1, st.session_state.idx - 1)

# ------------------------------------------------------------
#  UI
# ------------------------------------------------------------
st.title("🧠 G-code 3D Viewer (Play/Pause/Stop/Speed, E: 파란/회색 실선)")
uploaded_file = st.file_uploader("G-code 파일 업로드", type=["gcode", "nc"])

# 파일 업로드 & 파싱
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".gcode") as tmp:
        tmp.write(uploaded_file.read()); temp_path = tmp.name

    coords, is_extrudes, f_value = parse_gcode(temp_path)

    if len(coords) < 2:
        st.error("⚠ G-code 내 유효한 좌표가 부족합니다.")
        st.stop()
    else:
        # 상태 업데이트
        st.session_state.coords = coords
        st.session_state.is_extrudes = is_extrudes
        st.session_state.f_value = f_value
        st.session_state.idx = min(max(1, st.session_state.idx), len(coords) - 1)

        total_distance = compute_total_distance(coords)
        z_min, z_max = float(coords[:, 2].min()), float(coords[:, 2].max())
        if st.session_state.z_limit is None:
            st.session_state.z_limit = z_max

        est_time_min = (total_distance / f_value) if f_value > 0 else 0.0

        col_stats, col_ctrl = st.columns([2, 1])
        with col_stats:
            st.markdown(f"""
- 총 세그먼트 수: **{len(coords)-1}**
- 총 이동 거리: **{total_distance:.2f} mm**
- F값 (이송 속도): **{f_value:.1f} mm/min**
- 예상 소요 시간: **{est_time_min:.2f} 분**
- Z 범위: **{z_min:.1f} ~ {z_max:.1f} mm**
- 현재 표시 라인: **{st.session_state.idx}/{len(coords)-1}**
""")

        with col_ctrl:
            st.session_state.z_limit = st.slider(
                "🧱 Z 진행 높이 (mm)",
                min_value=z_min, max_value=z_max,
                value=float(st.session_state.z_limit), step=1.0
            )
            st.session_state.speed_lps = st.slider(
                "⏩ 재생 속도 (초당 라인 수)",
                min_value=1, max_value=500,
                value=int(st.session_state.speed_lps), step=1
            )
            st.session_state.loop_play = st.toggle("🔁 끝에서 루프 재생", value=bool(st.session_state.loop_play))

            c1, c2, c3 = st.columns(3)
            with c1: st.button("▶️ Play", use_container_width=True, on_click=on_play)
            with c2: st.button("⏸️ Pause", use_container_width=True, on_click=on_pause)
            with c3: st.button("⏹️ Stop", use_container_width=True, on_click=on_stop)

            c4, c5 = st.columns(2)
            with c4: st.button("⬅️ Step -1", use_container_width=True, on_click=step_backward)
            with c5: st.button("Step +1 ➡️", use_container_width=True, on_click=step_forward)

        # 그래프
        with st.expander("🔍 경로 시각화 보기", expanded=True):
            fig = plot_path_partial(coords, is_extrudes, st.session_state.idx, st.session_state.z_limit)
            st.plotly_chart(fig, use_container_width=True)

        # -------- 안정적인 재생 루프 --------
        # NOTE:
        # - 최신 Streamlit에서는 experimental_rerun 제거 → st.rerun() 사용
        # - 프레임 템포는 target_fps로 고정 (디스플레이 안정화)
        if st.session_state.playing:
            now = time.time()
            dt = now - st.session_state.last_tick if st.session_state.last_tick else 0.0
            min_step_interval = 1.0 / max(1.0, st.session_state.target_fps)
            if dt >= min_step_interval:
                # 프레임마다 진행할 라인 수
                lines_per_frame = max(1, int(st.session_state.speed_lps / st.session_state.target_fps))
                st.session_state.idx += lines_per_frame
                if st.session_state.idx >= len(coords) - 1:
                    if st.session_state.loop_play:
                        st.session_state.idx = 1
                    else:
                        st.session_state.idx = len(coords) - 1
                        st.session_state.playing = False
                st.session_state.last_tick = now
            # 살짝 쉬고 다시 렌더링
            time.sleep(1.0 / max(10.0, st.session_state.target_fps))  # CPU 과점유 방지
            st.rerun()

# ------------------------------------------------------------
#  사용 설명
# ------------------------------------------------------------
st.markdown("""
**📘 사용 방법**
1. `.gcode`/`.nc` 파일을 업로드합니다.
2. E값이 있는 압출 구간은 **파란색 실선**, 없는 이동 구간은 **얇은 회색 실선**입니다.
3. **Play / Pause / Stop**, **초당 라인 수**로 애니메이션을 제어합니다.
4. **Step ±1**로 1라인(선분)씩 수동 이동이 가능합니다.
5. **🔁 루프 재생**을 켜면 끝까지 간 뒤 처음부터 다시 재생합니다.
""")
