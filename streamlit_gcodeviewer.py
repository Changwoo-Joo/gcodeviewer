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
        is_extrude = None
        for axis, value in matches:
            axis_upper = axis.upper()
            if axis_upper in last_pos:
                last_pos[axis_upper] = float(value)
            elif axis_upper == 'F':
                f_value = float(value)
            elif axis_upper == 'E':
                is_extrude = float(value) > 0

        if None not in last_pos.values():
            coords_buffer.append([last_pos['X'], last_pos['Y'], last_pos['Z']])
            is_extrudes_buffer.append(is_extrude if is_extrude is not None else False)

        if idx % 1000 == 0 or idx == total_lines - 1:
            progress_bar.progress((idx + 1) / total_lines, text=f"🔄 파싱 중... {int((idx + 1) / total_lines * 100)}%")
            time.sleep(0.001)

    progress_bar.empty()

    coords = np.array(coords_buffer)
    is_extrudes = is_extrudes_buffer
    return coords, is_extrudes, f_value

# ------------------------------------------------------------
#  거리 계산 함수
# ------------------------------------------------------------
def compute_total_distance(coords):
    diffs = np.diff(coords, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return np.sum(distances)

# ------------------------------------------------------------
#  시각화 함수 (실선/점선 구분 + 선 연결 유지)
# ------------------------------------------------------------
def plot_path_by_z(coords, is_extrudes, max_z):
    def group_segments(coords, is_extrudes, target=True):
        group = []
        grouped_lines = []

        for i in range(1, len(coords)):
            if coords[i][2] > max_z and coords[i - 1][2] > max_z:
                continue  # 둘 다 범위 밖이면 무시

            is_match = is_extrudes[i] == target
            if is_match:
                group.append(coords[i - 1])
                group.append(coords[i])
            elif group:
                grouped_lines.append(np.array(group))
                group = []

        if group:
            grouped_lines.append(np.array(group))
        return grouped_lines

    fig = go.Figure()

    # 실선: 압출 구간
    ex_segments = group_segments(coords, is_extrudes, target=True)
    for seg in ex_segments:
        a = seg.T
        fig.add_trace(go.Scatter3d(
            x=a[0], y=a[1], z=a[2],
            mode='lines',
            line=dict(color='blue', width=3),
            showlegend=False
        ))

    # 점선: 비압출 이동 구간
    move_segments = group_segments(coords, is_extrudes, target=False)
    for seg in move_segments:
        a = seg.T
        fig.add_trace(go.Scatter3d(
            x=a[0], y=a[1], z=a[2],
            mode='lines',
            line=dict(color='gray', width=1, dash='dot'),
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
st.title("🧠 G-code 3D Viewer (속도 + 실선/점선 + 부드러운 연결)")

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
1. `.gcode` 또는 `.nc` 형식의 G-code 파일을 업로드하세요.
2. Z 높이에 따라 점진적으로 출력 경로가 시각화됩니다.
3. E값에 따라 실선(압출)과 점선(이동)이 구분되어 출력됩니다.
4. 문의: 동아로보틱스(주) 기술연구소 주창우 부장 (010-6754-2575)
""")
