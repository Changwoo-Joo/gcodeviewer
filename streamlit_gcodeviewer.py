import streamlit as st
import numpy as np
import re
import plotly.graph_objects as go
import tempfile
import chardet
import time

st.set_page_config(layout="wide")

# ------------------------------------------------------------
#  G-code 파싱 함수
# ------------------------------------------------------------
def parse_gcode(file_path):
    coords_buffer = []
    is_extrudes_buffer = []
    f_value = 0.0
    last_pos = {'X': None, 'Y': None, 'Z': None}
    current_extrude = False  # 기본값

    with open(file_path, 'rb') as raw_file:
        raw_data = raw_file.read()
        encoding = chardet.detect(raw_data)['encoding']

    lines = raw_data.decode(encoding, errors='replace').splitlines()
    total_lines = len(lines)
    progress_bar = st.progress(0, text="🔄 G-code 파싱 중...")

    for idx, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith(";"):
            continue

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

        if None not in last_pos.values():
            coords_buffer.append([last_pos['X'], last_pos['Y'], last_pos['Z']])
            is_extrudes_buffer.append(current_extrude if found_e else current_extrude)

        if idx % 1000 == 0 or idx == total_lines - 1:
            progress_bar.progress((idx + 1) / total_lines, text=f"🔄 파싱 중... {int((idx + 1) / total_lines * 100)}%")
            time.sleep(0.001)

    progress_bar.empty()

    coords = np.array(coords_buffer)
    # 🔧 E값이 전혀 없는 경우 → 전부 실선으로 처리
    if not any(is_extrudes_buffer):
        is_extrudes_buffer = [True] * len(coords_buffer)

    return coords, is_extrudes_buffer, f_value

# ------------------------------------------------------------
#  거리 계산 함수
# ------------------------------------------------------------
def compute_total_distance(coords):
    diffs = np.diff(coords, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return np.sum(distances)

# ------------------------------------------------------------
#  시각화 함수 (실선 처리만)
# ------------------------------------------------------------
def plot_path_by_z(coords, is_extrudes, max_z):
    def group_segments(coords, is_extrudes, target=True):
        group = []
        grouped_lines = []

        for i in range(1, len(coords)):
            if coords[i][2] > max_z and coords[i - 1][2] > max_z:
                continue

            if is_extrudes[i] == target:
                if not group:
                    group.append(coords[i - 1])
                group.append(coords[i])
            elif group:
                grouped_lines.append(np.array(group))
                group = []

        if group:
            grouped_lines.append(np.array(group))

        return grouped_lines

    fig = go.Figure()

    # 실선 처리 (회색 얇은 선)
    ex_segments = group_segments(coords, is_extrudes, target=True)
    for seg in ex_segments:
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
#  Streamlit 앱 UI
# ------------------------------------------------------------
st.title("🧠 G-code 3D Viewer (E값 없으면 회색 실선)")

uploaded_file = st.file_uploader("G-code 파일 업로드", type=["gcode", "nc"])

if uploaded_file:
    st.info("🧠 G-code 파일 업로드 완료. 파싱을 시작합니다...")
    time.sleep(1)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".gcode") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    coords, is_extrudes, f_value = parse_gcode(temp_path)

    if len(coords) < 2:
        st.error("⚠ G-code 내 유효한 좌표가 부족합니다.")
    else:
        total_distance = compute_total_distance(coords)
        est_time = total_distance / f_value if f_value > 0 else 0
        z_min, z_max = float(coords[:,2].min()), float(coords[:,2].max())

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            - 총 세그먼트 수: **{len(coords)-1}**
            - 총 이동 거리: **{total_distance:.2f} mm**
            - F값 (이송 속도): **{f_value:.1f} mm/min**
            - 예상 소요 시간: **{est_time:.2f} 분**
            - Z 범위: **{z_min:.1f} ~ {z_max:.1f} mm**
            """)

        with col2:
            current_z = st.slider("🧱 Z 진행 높이 (mm)", min_value=z_min, max_value=z_max, value=z_max, step=1.0)

        with st.expander("🔍 경로 시각화 보기", expanded=True):
            fig = plot_path_by_z(coords, is_extrudes, current_z)
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
#  사용 설명
# ------------------------------------------------------------
st.markdown("""
**📘 사용 방법**
1. `.gcode` 또는 `.nc` 파일을 업로드하면 G-code를 시각화합니다.
2. E값이 포함된 경우 실선/점선을 구분하고, 없으면 모두 회색 실선으로 처리합니다.
3. Z 슬라이더를 통해 높이별 경로를 필터링할 수 있습니다.
""")
