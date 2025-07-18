import streamlit as st
import numpy as np
import re
import plotly.graph_objects as go
import tempfile

def parse_gcode(file_path):
    coords = []
    is_extrudes = []
    f_value = 0
    last_pos = {'X': None, 'Y': None, 'Z': None}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            x = re.search(r'[Xx]([-+]?[0-9]*\.?[0-9]+)', line)
            y = re.search(r'[Yy]([-+]?[0-9]*\.?[0-9]+)', line)
            z = re.search(r'[Zz]([-+]?[0-9]*\.?[0-9]+)', line)
            e = re.search(r'[Ee]([-+]?[0-9]*\.?[0-9]+)', line)
            f = re.search(r'[Ff]([-+]?[0-9]*\.?[0-9]+)', line)
            if x:
                last_pos['X'] = float(x.group(1))
            if y:
                last_pos['Y'] = float(y.group(1))
            if z:
                last_pos['Z'] = float(z.group(1))
            if f:
                f_value = float(f.group(1))
            if None not in last_pos.values():
                coords.append([last_pos['X'], last_pos['Y'], last_pos['Z']])
                is_extrudes.append(float(e.group(1)) > 0 if e else False)
    return np.array(coords), is_extrudes, f_value

def compute_total_distance(coords):
    diffs = np.diff(coords, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return np.sum(distances)

def plot_path(coords, is_extrudes, fraction):
    fig = go.Figure()
    n = int(len(coords) * fraction)
    for i in range(1, n):
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
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        width=1000,
        height=700
    )
    return fig

st.title("🧠 G-code 3D 시각화 (Plotly 기반)")

uploaded_file = st.file_uploader("G-code 파일 업로드", type=["gcode", "nc"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".gcode") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    coords, is_extrudes, f_value = parse_gcode(temp_path)

    if len(coords) < 2:
        st.error("G-code 내 유효한 좌표가 부족합니다.")
    else:
        total_distance = compute_total_distance(coords)
        est_time = total_distance / f_value if f_value > 0 else 0

        st.success("✅ G-code 분석 완료")
        st.markdown(f"""
        - 총 세그먼트 수: **{len(coords)-1}**
        - 총 이동 거리: **{total_distance:.2f} mm**
        - F값 (이송 속도): **{f_value:.1f} mm/min**
        - 예상 소요 시간: **{est_time:.2f} 분**
        """)

        progress = st.slider("경로 표시 비율 (%)", min_value=1, max_value=100, value=100)
        fraction = progress / 100.0

        fig = plot_path(coords, is_extrudes, fraction)
        st.plotly_chart(fig, use_container_width=True)
