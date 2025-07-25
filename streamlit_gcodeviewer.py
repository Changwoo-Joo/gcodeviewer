import streamlit as st
import numpy as np
import re
import plotly.graph_objects as go
import tempfile
import chardet
import time

st.set_page_config(layout="wide")

# ------------------------------------------------------------
#  최적화된 G-code 파싱 함수
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
#  Z 기준 경로 시각화
# ------------------------------------------------------------
def plot_path_by_z(coords, is_extrudes, max_z):
    fig = go.Figure()
    for i in range(1, len(coords)):
        if coords[i][2] <= max_z or coords[i-1][2] <= max_z:
            x, y, z = zip(coords[i-1], coords[i])
            color = 'blue' if is_extrudes[i] else 'gray'
            width = 4 if is_extrudes[i] else 2
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color=color, width=width),
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
#  Streamlit 앱 UI 구성
# ------------------------------------------------------------
st.title("🧠 G-code 3D Viewer (고속 최적화 버전)")

uploaded_file = st.file_uploader("G-code 파일 업로드", type=["gcode", "nc"])

if uploaded_file:
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
2. 분석 결과와 함께 Z 높이에 따라 점진적으로 출력 경로가 시각화됩니다.
3. 슬라이더를 움직여 Z 방향으로 경로가 쌓이는 과정을 확인하세요.
4. 문의: 동아로보틱스(주) 기술연구소 주창우부장(010-6754-2575)
""")
