import streamlit as st
import numpy as np
import re
import plotly.graph_objects as go
import tempfile
import chardet

st.set_page_config(layout="wide")

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
        enc_info = chardet.detect(raw_data)
        encoding = enc_info.get('encoding') or 'utf-8'  # 인코딩 기본값 가드
        lines = raw_data.decode(encoding, errors='replace').splitlines()

    for line in lines:
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

    coords = np.array(coords_buffer)
    # E값이 모두 없는 경우 → False만 존재 → 회색 실선 처리용으로 유지
    return coords, is_extrudes_buffer, f_value


# ------------------------------------------------------------
#  거리 계산 함수
# ------------------------------------------------------------
def compute_total_distance(coords):
    if len(coords) < 2:
        return 0.0
    diffs = np.diff(coords, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return np.sum(distances)


# ------------------------------------------------------------
#  시각화 함수 (E값 여부로 스타일 분기 + 연결 유지)
# ------------------------------------------------------------
def plot_path_by_z(coords, is_extrudes, max_z):
    def group_segments(coords, is_extrudes, target_extrude):
        group = []
        grouped_lines = []
        for i in range(1, len(coords)):
            # max_z 초과하는 두 점 사이의 선분은 제외
            if coords[i][2] > max_z and coords[i - 1][2] > max_z:
                continue
            if is_extrudes[i] == target_extrude:
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

    # 🔵 파란 실선 (E값 있음)
    ex_segments = group_segments(coords, is_extrudes, target_extrude=True)
    for seg in ex_segments:
        a = seg.T
        fig.add_trace(go.Scatter3d(
            x=a[0], y=a[1], z=a[2],
            mode='lines',
            line=dict(color='blue', width=2),
            showlegend=False
        ))

    # ⚪ 회색 실선 (E값 없음 → 이동만)
    move_segments = group_segments(coords, is_extrudes, target_extrude=False)
    for seg in move_segments:
        a = seg.T
        fig.add_trace(go.Scatter3d(
            x=a[0], y=a[1], z=a[2],
            mode='lines',
            line=dict(color='gray', width=1),
            showlegend=False
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=700
    )
    return fig


# ------------------------------------------------------------
#  Streamlit UI
# ------------------------------------------------------------
st.title("🧠 G-code 3D Viewer (E값 구분: 파란/회색 실선)")

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
        z_min, z_max = float(coords[:, 2].min()), float(coords[:, 2].max())

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
            if z_min == z_max:
                current_z = z_max
                st.info(f"단일 레이어(Z={z_max:.1f} mm)입니다.")
            else:
                current_z = st.slider(
                    "🧱 Z 진행 높이 (mm)",
                    min_value=z_min,
                    max_value=z_max,
                    value=z_max,
                    step=1.0
                )

        with st.expander("🔍 경로 시각화 보기", expanded=True):
            fig = plot_path_by_z(coords, is_extrudes, current_z)
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
#  사용 설명
# ------------------------------------------------------------
st.markdown("""
**📘 사용 방법**

1. .gcode 또는 .nc 형식의 G-code 파일을 업로드하세요.  
2. E값이 있는 압출 구간은 **파란색 실선**, 없는 이동 구간은 **얇은 회색 실선**으로 구분됩니다.  
3. Z 슬라이더로 출력 높이를 제한하며 경로를 확인할 수 있습니다.  
""")
