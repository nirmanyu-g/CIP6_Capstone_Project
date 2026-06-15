import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Session State Initialization ---
if 'supports' not in st.session_state:
    st.session_state.supports = []
if 'loads' not in st.session_state:
    st.session_state.loads = []

def run_fem_analysis(length, E, G, area, inertia, k_factor, theory, elements, supports, loads):
    """Runs the matrix math and returns the matplotlib figure."""
    N = elements
    total_nodes = N + 1
    total_dofs = 2 * total_nodes
    L_e = length / N
    
    K_global = np.zeros((total_dofs, total_dofs))
    F_global = np.zeros(total_dofs)
    
    EI = E * inertia
    phi = (12 * EI) / (k_factor * G * area * (L_e ** 2)) if theory == "Timoshenko" else 0.0
        
    denom = (L_e ** 3) * (1 + phi)
    ke = (EI / denom) * np.array([
        [12,            6*L_e,          -12,           6*L_e],
        [6*L_e,         (4+phi)*L_e**2, -6*L_e,        (2-phi)*L_e**2],
        [-12,           -6*L_e,         12,            -6*L_e],
        [6*L_e,         (2-phi)*L_e**2, -6*L_e,        (4+phi)*L_e**2]
    ])
    
    for i in range(N):
        dofs = [2*i, 2*i+1, 2*i+2, 2*i+3]
        for r in range(4):
            for c in range(4):
                K_global[dofs[r], dofs[c]] += ke[r, c]
                
    for l in loads:
        if l["type"] == "Point":
            node_idx = int(round((l["pos"] / length) * N))
            F_global[2 * node_idx] -= l["mag"]
        elif l["type"] == "Moment":
            node_idx = int(round((l["pos"] / length) * N))
            F_global[2 * node_idx + 1] += l["mag"]
        elif l["type"] == "Dist":
            xa, xb = l["start"], l["end"]
            wa, wb = l["w1"], l["w2"]
            for i in range(N):
                x_left = i * L_e
                x_right = (i + 1) * L_e
                start_x = max(x_left, xa)
                end_x = min(x_right, xb)
                if start_x < end_x:
                    w_s = wa + (wb - wa) * (start_x - xa) / (xb - xa) if xb != xa else wa
                    w_e = wa + (wb - wa) * (end_x - xa) / (xb - xa) if xb != xa else wa
                    total_force = (w_s + w_e) / 2.0 * (end_x - start_x)
                    F_global[2 * i] -= total_force / 2.0
                    F_global[2 * i + 2] -= total_force / 2.0
        
    constrained_dofs = set()
    for s in supports:
        node_idx = int(round((s["pos"] / length) * N))
        dof_v = 2 * node_idx
        dof_t = 2 * node_idx + 1
        
        if s["type"] == "Fixed":
            constrained_dofs.update([dof_v, dof_t])
        elif s["type"] == "Pin":
            constrained_dofs.add(dof_v)
        elif s["type"] == "Guided":
            constrained_dofs.add(dof_t)
        elif s["type"] == "Spring":
            K_global[dof_v, dof_v] += s["kv"]
            K_global[dof_t, dof_t] += s["kt"]
            
    free_dofs = [dof for dof in range(total_dofs) if dof not in constrained_dofs]
    
    try:
        K_free = K_global[np.ix_(free_dofs, free_dofs)]
        F_free = F_global[free_dofs]
        U_free = np.linalg.solve(K_free, F_free)
    except np.linalg.LinAlgError:
        return None, "Matrix singular. The structure is mathematically unstable."
        
    U_global = np.zeros(total_dofs)
    U_global[free_dofs] = U_free
    
    x_plot = np.linspace(0, length, total_nodes)
    V_plot = np.zeros(total_nodes)
    M_plot = np.zeros(total_nodes)
    
    for i in range(N):
        u_e = U_global[[2*i, 2*i+1, 2*i+2, 2*i+3]]
        f_e = np.dot(ke, u_e)
        V_plot[i] = f_e[0]
        M_plot[i] = -f_e[1]
    V_plot[-1] = -f_e[2]
    M_plot[-1] = f_e[3]
    
    deflection_mm = U_global[0::2] * 1000

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 14))
    fig.tight_layout(pad=6.0)
    
    ax1.set_title(f"FEM Configuration ({theory})")
    ax1.plot([0, length], [0, 0], color='black', linewidth=4)
    for s in supports:
        if s["type"] == "Fixed": ax1.plot(s["pos"], 0, 's', markersize=15, color='purple')
        elif s["type"] == "Pin": ax1.plot(s["pos"], 0, '^', markersize=15, color='blue')
        elif s["type"] == "Guided": ax1.plot(s["pos"], 0, 'D', markersize=12, color='cyan')
        elif s["type"] == "Spring": ax1.plot(s["pos"], 0, 'o', markersize=12, color='green')
        
    for l in loads:
        if l["type"] == "Point":
            ax1.annotate(f"{l['mag']/1000:.1f}kN", xy=(l["pos"], 0), xytext=(l["pos"], 1.5),
                         arrowprops=dict(facecolor='red', shrink=0.05), ha='center', color='red')
        elif l["type"] == "Moment":
            ax1.plot(l["pos"], 0.8, 'o', color='orange', markersize=10)
            ax1.text(l["pos"], 1.2, f"{l['mag']/1000:.1f}kNm ↺", ha='center', color='orange')
        elif l["type"] == "Dist":
            xa, xb = l["start"], l["end"]
            max_w = max(abs(l["w1"]), abs(l["w2"])) or 1
            h1, h2 = (abs(l["w1"]) / max_w), (abs(l["w2"]) / max_w)
            ax1.fill_between([xa, xb], [h1, h2], 0, alpha=0.3, color='red')
            ax1.text((xa+xb)/2, max(h1, h2) + 0.3, f"UDL: {l['w1']/1000:.1f} to {l['w2']/1000:.1f}", ha='center', color='red')
            
    ax1.set_xlim(-0.5, length + 0.5)
    ax1.set_ylim(-1, 3)
    ax1.axis('off')
    
    ax2.set_title("Shear Force Diagram (SFD)")
    ax2.plot(x_plot, V_plot / 1000, color='blue')
    ax2.fill_between(x_plot, V_plot / 1000, 0, alpha=0.3, color='blue')
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_ylabel("Shear (kN)")
    ax2.grid(True, linestyle='--')
    
    ax3.set_title("Bending Moment Diagram (BMD)")
    ax3.plot(x_plot, M_plot / 1000, color='red')
    ax3.fill_between(x_plot, M_plot / 1000, 0, alpha=0.3, color='red')
    ax3.axhline(0, color='black', linewidth=1)
    ax3.set_ylabel("Moment (kNm)")
    ax3.grid(True, linestyle='--')
    
    ax4.set_title("Deflection Profile Line")
    ax4.plot(x_plot, deflection_mm, color='green', linewidth=2)
    ax4.fill_between(x_plot, deflection_mm, 0, alpha=0.3, color='green')
    ax4.axhline(0, color='black', linewidth=1)
    ax4.invert_yaxis()
    ax4.set_xlabel("Distance (m)")
    ax4.set_ylabel("Deflection (mm)")
    ax4.grid(True, linestyle='--')
    
    stats = {
        "shear": np.max(np.abs(V_plot))/1000,
        "moment": np.max(np.abs(M_plot))/1000,
        "deflection": np.max(np.abs(deflection_mm))
    }
    return fig, stats

# --- UI Layout ---
st.set_page_config(page_title="FEM Beam App", layout="wide")
st.title("Structural Beam Analysis App")

# Sidebar Configuration
with st.sidebar:
    st.header("Global Settings")
    length = st.number_input("Beam Length (m)", value=25.0, min_value=0.1)
    elements = st.number_input("Math Elements", value=200, min_value=10, max_value=1000)
    theory = st.selectbox("Beam Theory", ["Euler-Bernoulli", "Timoshenko"])
    
    st.header("Material (GPa)")
    E_gpa = st.number_input("Young's Modulus (E)", value=200.0)
    G_gpa = st.number_input("Shear Modulus (G)", value=50.0)
    
    st.header("Cross Section")
    area = st.number_input("Area (m^2)", value=25.0)
    inertia = st.number_input("Moment of Inertia (m^4)", value=32.0)
    k_factor = st.number_input("Shear Factor (k)", value=0.8)

# Main Screen Columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Boundary Conditions")
    sup_type = st.selectbox("Support Type", ["Fixed", "Pin", "Guided", "Spring"])
    sup_pos = st.number_input("Support Position (m)", min_value=0.0, max_value=float(length), value=0.0)
    
    kv, kt = 0.0, 0.0
    if sup_type == "Spring":
        kv = st.number_input("Vertical Stiffness (N/m)", value=1000.0)
        kt = st.number_input("Rotational Stiffness (Nm/rad)", value=1000.0)
        
    if st.button("Add Support"):
        st.session_state.supports.append({"type": sup_type, "pos": sup_pos, "kv": kv, "kt": kt})
        
    if st.button("Clear Supports"):
        st.session_state.supports = []
        
    st.write(st.session_state.supports)

with col2:
    st.subheader("External Loads")
    load_type = st.selectbox("Load Type", ["Point", "Moment", "Distributed"])
    
    if load_type in ["Point", "Moment"]:
        mag = st.number_input("Magnitude (kN or kNm)", value=10.0)
        l_pos = st.number_input("Load Position (m)", min_value=0.0, max_value=float(length), value=length/2)
        if st.button("Add Load"):
            st.session_state.loads.append({"type": load_type, "mag": mag * 1000, "pos": l_pos})
    else:
        l_start = st.number_input("Start Position (m)", value=0.0)
        l_end = st.number_input("End Position (m)", value=length)
        w1 = st.number_input("Start Mag (kN/m)", value=10.0)
        w2 = st.number_input("End Mag (kN/m)", value=10.0)
        if st.button("Add Distributed Load"):
            st.session_state.loads.append({"type": "Dist", "start": l_start, "end": l_end, "w1": w1 * 1000, "w2": w2 * 1000})

    if st.button("Clear Loads"):
        st.session_state.loads = []
        
    st.write(st.session_state.loads)

st.divider()

# Analysis Execution
if st.button("Run FEM Analysis", type="primary"):
    if not st.session_state.supports:
        st.error("Structure is unstable. Add at least one support constraint.")
    else:
        with st.spinner("Calculating matrix..."):
            fig, result = run_fem_analysis(length, E_gpa * 1e9, G_gpa * 1e9, area, inertia, k_factor, theory, elements, st.session_state.supports, st.session_state.loads)
            
            if fig is None:
                st.error(result) # Show instability error
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Max Shear", f"{result['shear']:.2f} kN")
                c2.metric("Max Moment", f"{result['moment']:.2f} kNm")
                c3.metric("Max Deflection", f"{result['deflection']:.2f} mm")
                st.pyplot(fig)
