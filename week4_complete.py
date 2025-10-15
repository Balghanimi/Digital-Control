# ═══════════════════════════════════════════════════════════════════
# WEEK 4: Z-TRANSFORM INTERACTIVE DEMONSTRATIONS
# Digital Control Systems | University of Kufa
# Dr. Ali Al-Ghanimi | Electrical Engineering Department
# ═══════════════════════════════════════════════════════════════════
# Sources: lec 2.pdf, lec 4.pdf, notes_A2_DiscreteSystems.pdf

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import control as ct
from io import BytesIO

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Week 4: Z-Transform | Digital Control",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════
# CUSTOM STYLING
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .demo-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .result-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .success-box {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #22c55e;
        margin: 1rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #ffe259 0%, #ffa751 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
    }
    h1, h2, h3 {
        font-family: 'Arial', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
    <div class="main-header">
        <h1>🎓 Digital Control Systems</h1>
        <h2>Week 4: Z-Transform & Inverse Z-Transform</h2>
        <p style="font-size: 1.1rem; margin-top: 1rem;">
            <strong>Dr. Ali Al-Ghanimi</strong> | Electrical Engineering Department<br>
            University of Kufa
        </p>
    </div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 📚 Select Demo")
    demo_choice = st.selectbox(
        "Choose an interactive demonstration:",
        [
            "📊 Analytics Dashboard",
            "🎯 Demo 1: Z-Transform Definition",
            "🔄 Demo 2: Region of Convergence (ROC)",
            "⚡ Demo 3: Inverse Z-Transform Methods",
            "🧮 Demo 4: Partial Fraction Expansion",
            "📈 Demo 5: Transform Properties"
        ]
    )
    
    st.markdown("---")
    st.markdown("### 🎯 Learning Objectives")
    st.markdown("""
    - ✅ Z-transform definition
    - ✅ ROC visualization
    - ✅ Inverse z-transform methods
    - ✅ Partial fraction expansion
    - ✅ Transform properties
    """)
    
    st.markdown("---")
    st.markdown("### 📖 Course Materials")
    st.markdown("""
    - 📄 Lecture 2 (Slides)
    - 📄 Lecture 4 (Examples)
    - 📘 Chakrabortty et al.
    - 📗 Notes: Discrete Systems
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Need Help?")
    st.info("Use the interactive sliders to explore different scenarios. Each demo includes explanations and visualizations.")

# ═══════════════════════════════════════════════════════════════════
# ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════════════

if demo_choice == "📊 Analytics Dashboard":
    st.markdown('<div class="demo-card">', unsafe_allow_html=True)
    st.markdown("## 📊 Usage Analytics Dashboard")
    st.markdown("### Track Student Engagement")
    st.markdown('</div>', unsafe_allow_html=True)
    
    password = st.text_input("Enter Admin Password:", type="password")
    
    if password == "kufa2025":
        st.success("✅ Access Granted!")
        
        # Initialize session state for analytics
        if 'analytics' not in st.session_state:
            st.session_state.analytics = {
                'total_views': 0,
                'demo1_views': 0,
                'demo2_views': 0,
                'demo3_views': 0,
                'demo4_views': 0,
                'demo5_views': 0
            }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📈 Total Views", st.session_state.analytics['total_views'])
        with col2:
            st.metric("🎯 Demo 1 Views", st.session_state.analytics['demo1_views'])
        with col3:
            st.metric("🔄 Demo 2 Views", st.session_state.analytics['demo2_views'])
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            st.metric("⚡ Demo 3 Views", st.session_state.analytics['demo3_views'])
        with col5:
            st.metric("🧮 Demo 4 Views", st.session_state.analytics['demo4_views'])
        with col6:
            st.metric("📈 Demo 5 Views", st.session_state.analytics['demo5_views'])
        
        # Visualization
        st.markdown("### 📊 Usage Distribution")
        demo_names = ['Demo 1', 'Demo 2', 'Demo 3', 'Demo 4', 'Demo 5']
        demo_counts = [
            st.session_state.analytics['demo1_views'],
            st.session_state.analytics['demo2_views'],
            st.session_state.analytics['demo3_views'],
            st.session_state.analytics['demo4_views'],
            st.session_state.analytics['demo5_views']
        ]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(demo_names, demo_counts, color='#667eea')
        ax.set_ylabel('Number of Views')
        ax.set_title('Demo Usage Statistics')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
        
    elif password:
        st.error("❌ Incorrect password!")

# ═══════════════════════════════════════════════════════════════════
# DEMO 1: Z-TRANSFORM DEFINITION
# ═══════════════════════════════════════════════════════════════════

elif demo_choice == "🎯 Demo 1: Z-Transform Definition":
    # Track view
    if 'analytics' not in st.session_state:
        st.session_state.analytics = {'demo1_views': 0, 'total_views': 0}
    st.session_state.analytics['demo1_views'] += 1
    st.session_state.analytics['total_views'] += 1
    
    st.markdown('<div class="demo-card">', unsafe_allow_html=True)
    st.markdown("## 🎯 Demo 1: Z-Transform Definition")
    st.markdown("**Objective:** Visualize the z-transform of basic sequences")
    st.markdown("**Source:** lec 2.pdf, p.5-8")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### 📐 Theory")
    st.latex(r"X(z) = \sum_{n=0}^{\infty} x[n] z^{-n}")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Parameters")
        seq_type = st.selectbox("Sequence Type", 
                                 ["Unit Step", "Unit Impulse", "Exponential", "Ramp"])
        
        if seq_type == "Exponential":
            a = st.slider("Coefficient a", 0.1, 2.0, 0.8, 0.1)
        
        n_points = st.slider("Number of Points", 5, 20, 10)
    
    with col2:
        n = np.arange(0, n_points)
        
        if seq_type == "Unit Step":
            x = np.ones(n_points)
            st.latex(r"x[n] = u[n]")
            st.latex(r"X(z) = \frac{z}{z-1}, \quad |z| > 1")
        elif seq_type == "Unit Impulse":
            x = np.zeros(n_points)
            x[0] = 1
            st.latex(r"x[n] = \delta[n]")
            st.latex(r"X(z) = 1")
        elif seq_type == "Exponential":
            x = a**n
            st.latex(f"x[n] = {a}^n")
            st.latex(f"X(z) = \\frac{{z}}{{z-{a}}}, \\quad |z| > {a}")
        else:  # Ramp
            x = n
            st.latex(r"x[n] = n")
            st.latex(r"X(z) = \frac{z}{(z-1)^2}, \quad |z| > 1")
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.stem(n, x, basefmt=' ', linefmt='C0-', markerfmt='C0o')
        ax.set_xlabel('n')
        ax.set_ylabel('x[n]')
        ax.set_title(f'{seq_type} Sequence')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Download
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        st.download_button("📥 Download Plot", buf.getvalue(), 
                          f"ztransform_{seq_type}.png", "image/png")
    
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.markdown("✅ **Key Insight:** The z-transform converts discrete-time sequences into complex functions, enabling frequency-domain analysis.")
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# DEMO 2: REGION OF CONVERGENCE (ROC)
# ═══════════════════════════════════════════════════════════════════

elif demo_choice == "🔄 Demo 2: Region of Convergence (ROC)":
    # Track view
    if 'analytics' not in st.session_state:
        st.session_state.analytics = {'demo2_views': 0, 'total_views': 0}
    st.session_state.analytics['demo2_views'] += 1
    st.session_state.analytics['total_views'] += 1
    
    st.markdown('<div class="demo-card">', unsafe_allow_html=True)
    st.markdown("## 🔄 Demo 2: Region of Convergence (ROC)")
    st.markdown("**Objective:** Visualize ROC in the z-plane")
    st.markdown("**Source:** lec 2.pdf, p.12-15")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Pole Locations")
        pole1_r = st.slider("Pole 1 Radius", 0.1, 2.0, 0.5, 0.1)
        pole1_angle = st.slider("Pole 1 Angle (°)", 0, 180, 30)
        
        pole2_r = st.slider("Pole 2 Radius", 0.1, 2.0, 0.8, 0.1)
        pole2_angle = st.slider("Pole 2 Angle (°)", 0, 180, 60)
    
    with col2:
        # Convert to complex
        pole1 = pole1_r * np.exp(1j * np.deg2rad(pole1_angle))
        pole2 = pole2_r * np.exp(1j * np.deg2rad(pole2_angle))
        poles = [pole1, np.conj(pole1), pole2, np.conj(pole2)]
        
        # Plot z-plane
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=2, label='Unit Circle')
        
        # Poles
        for pole in poles:
            ax.plot(pole.real, pole.imag, 'rx', markersize=15, markeredgewidth=3)
        
        # ROC (shaded region)
        max_radius = max(abs(pole1), abs(pole2))
        roc_outer = plt.Circle((0, 0), 2.5, color='green', alpha=0.1)
        roc_inner = plt.Circle((0, 0), max_radius, color='white', alpha=0.8)
        ax.add_patch(roc_outer)
        ax.add_patch(roc_inner)
        
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.set_title('Z-Plane: Poles (×) and ROC (Green Region)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.legend()
        st.pyplot(fig)
        
        st.markdown(f"**ROC:** |z| > {max_radius:.2f}")
        
        if max_radius < 1:
            st.success("✅ System is **STABLE** (poles inside unit circle)")
        else:
            st.warning("⚠️ System is **UNSTABLE** (poles outside unit circle)")

# ═══════════════════════════════════════════════════════════════════
# DEMO 3: INVERSE Z-TRANSFORM
# ═══════════════════════════════════════════════════════════════════

elif demo_choice == "⚡ Demo 3: Inverse Z-Transform Methods":
    # Track view
    if 'analytics' not in st.session_state:
        st.session_state.analytics = {'demo3_views': 0, 'total_views': 0}
    st.session_state.analytics['demo3_views'] += 1
    st.session_state.analytics['total_views'] += 1
    
    st.markdown('<div class="demo-card">', unsafe_allow_html=True)
    st.markdown("## ⚡ Demo 3: Inverse Z-Transform Methods")
    st.markdown("**Objective:** Compare long division vs partial fractions")
    st.markdown("**Source:** lec 4.pdf, Examples 1-3")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Transfer Function")
        st.latex(r"H(z) = \frac{z}{z^2 - 1.5z + 0.5}")
        
        method = st.radio("Method", ["Long Division", "Partial Fractions"])
        n_terms = st.slider("Number of Terms", 5, 15, 10)
    
    with col2:
        # Define system
        num = [1, 0]
        den = [1, -1.5, 0.5]
        
        if method == "Long Division":
            # Impulse response
            n = np.arange(n_terms)
            sys = signal.dlti(num, den, dt=1)
            _, h = signal.dimpulse(sys, n=n_terms)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.stem(n, h[0].flatten(), basefmt=' ')
            ax.set_xlabel('n')
            ax.set_ylabel('h[n]')
            ax.set_title('Impulse Response (Long Division Method)')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
        else:  # Partial Fractions
            st.markdown("### Step 1: Factor Denominator")
            st.latex(r"z^2 - 1.5z + 0.5 = (z-1)(z-0.5)")
            
            st.markdown("### Step 2: Partial Fractions")
            st.latex(r"\frac{z}{(z-1)(z-0.5)} = \frac{A}{z-1} + \frac{B}{z-0.5}")
            
            st.markdown("### Step 3: Solve for A, B")
            st.latex(r"A = 2, \quad B = -1")
            
            st.markdown("### Step 4: Inverse Transform")
            st.latex(r"h[n] = 2(1)^n - (-1)(0.5)^n = 2 - (0.5)^n")
            
            # Plot
            n = np.arange(n_terms)
            h = 2 - 0.5**n
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.stem(n, h, basefmt=' ')
            ax.set_xlabel('n')
            ax.set_ylabel('h[n]')
            ax.set_title('Impulse Response (Partial Fractions Method)')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

# ═══════════════════════════════════════════════════════════════════
# DEMO 4: PARTIAL FRACTION EXPANSION
# ═══════════════════════════════════════════════════════════════════

elif demo_choice == "🧮 Demo 4: Partial Fraction Expansion":
    # Track view
    if 'analytics' not in st.session_state:
        st.session_state.analytics = {'demo4_views': 0, 'total_views': 0}
    st.session_state.analytics['demo4_views'] += 1
    st.session_state.analytics['total_views'] += 1
    
    st.markdown('<div class="demo-card">', unsafe_allow_html=True)
    st.markdown("## 🧮 Demo 4: Partial Fraction Expansion")
    st.markdown("**Objective:** Interactive partial fraction calculator")
    st.markdown("**Source:** lec 4.pdf, p.8-12")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Define X(z)/z")
        
        # Numerator
        st.markdown("**Numerator:**")
        b0 = st.number_input("b₀", value=1.0)
        b1 = st.number_input("b₁", value=0.0)
        
        # Denominator
        st.markdown("**Denominator:**")
        a1 = st.number_input("a₁", value=-1.5)
        a2 = st.number_input("a₂", value=0.5)
    
    with col2:
        num = [b0, b1]
        den = [1, a1, a2]
        
        # Calculate poles
        poles = np.roots(den)
        
        st.markdown("### 📊 Results")
        
        st.markdown(f"**Transfer Function:**")
        st.latex(f"\\frac{{X(z)}}{{z}} = \\frac{{{b0}z + {b1}}}{{z^2 + ({a1})z + {a2}}}")
        
        st.markdown(f"**Poles:** {poles[0]:.3f}, {poles[1]:.3f}")
        
        # Partial fractions
        try:
            r, p, k = signal.residue(num, den)
            
            st.markdown("**Partial Fraction Form:**")
            st.latex(f"\\frac{{X(z)}}{{z}} = \\frac{{{r[0]:.3f}}}{{z-{p[0]:.3f}}} + \\frac{{{r[1]:.3f}}}{{z-{p[1]:.3f}}}")
            
            # Time domain
            n = np.arange(20)
            x = r[0] * p[0]**n + r[1] * p[1]**n
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.stem(n, x, basefmt=' ')
            ax.set_xlabel('n')
            ax.set_ylabel('x[n]')
            ax.set_title('Time-Domain Sequence')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
        except:
            st.error("⚠️ Cannot compute partial fractions for these coefficients")

# ═══════════════════════════════════════════════════════════════════
# DEMO 5: TRANSFORM PROPERTIES
# ═══════════════════════════════════════════════════════════════════

elif demo_choice == "📈 Demo 5: Transform Properties":
    # Track view
    if 'analytics' not in st.session_state:
        st.session_state.analytics = {'demo5_views': 0, 'total_views': 0}
    st.session_state.analytics['demo5_views'] += 1
    st.session_state.analytics['total_views'] += 1
    
    st.markdown('<div class="demo-card">', unsafe_allow_html=True)
    st.markdown("## 📈 Demo 5: Transform Properties")
    st.markdown("**Objective:** Visualize time-shifting, scaling, and convolution")
    st.markdown("**Source:** lec 2.pdf, p.18-25")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Select Property")
        property_type = st.selectbox("Property", 
                                      ["Time Shifting", "Time Scaling", "Convolution"])
        
        if property_type == "Time Shifting":
            k = st.slider("Shift k", -5, 5, 2)
        elif property_type == "Time Scaling":
            a = st.slider("Scale factor a", 0.5, 2.0, 1.5, 0.1)
        else:
            st.markdown("**Two sequences to convolve**")
    
    with col2:
        n = np.arange(15)
        
        if property_type == "Time Shifting":
            x = np.sin(n * 0.5)
            x_shifted = np.zeros_like(x)
            if k >= 0:
                x_shifted[k:] = x[:-k] if k > 0 else x
            else:
                x_shifted[:k] = x[-k:]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            ax1.stem(n, x, basefmt=' ')
            ax1.set_title('Original: x[n]')
            ax1.set_ylabel('x[n]')
            ax1.grid(True, alpha=0.3)
            
            ax2.stem(n, x_shifted, basefmt=' ')
            ax2.set_title(f'Shifted: x[n-{k}]')
            ax2.set_xlabel('n')
            ax2.set_ylabel(f'x[n-{k}]')
            ax2.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            st.latex(f"x[n-{k}] \\leftrightarrow X(z)z^{{-{k}}}")
            
        elif property_type == "Time Scaling":
            x = 0.8**n
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.stem(n, x, label='x[n] = 0.8ⁿ', basefmt=' ')
            ax.stem(n, a**n * x, label=f'aⁿx[n], a={a}', basefmt=' ')
            ax.set_xlabel('n')
            ax.set_ylabel('Amplitude')
            ax.set_title('Time Scaling Property')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.latex(f"a^n x[n] \\leftrightarrow X(z/a)")
            
        else:  # Convolution
            x1 = np.array([1, 2, 3, 2, 1, 0, 0, 0, 0, 0])
            x2 = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
            y = np.convolve(x1, x2)[:10]
            
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
            
            ax1.stem(range(10), x1, basefmt=' ')
            ax1.set_title('Sequence 1: x₁[n]')
            ax1.set_ylabel('x₁[n]')
            ax1.grid(True, alpha=0.3)
            
            ax2.stem(range(10), x2, basefmt=' ')
            ax2.set_title('Sequence 2: x₂[n]')
            ax2.set_ylabel('x₂[n]')
            ax2.grid(True, alpha=0.3)
            
            ax3.stem(range(10), y, basefmt=' ')
            ax3.set_title('Convolution: y[n] = x₁[n] * x₂[n]')
            ax3.set_xlabel('n')
            ax3.set_ylabel('y[n]')
            ax3.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            st.latex(r"x_1[n] * x_2[n] \leftrightarrow X_1(z) \cdot X_2(z)")

# ═══════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <h4>Digital Control Systems | Fall 2025</h4>
        <p><strong>Dr. Ali Al-Ghanimi</strong> | Electrical Engineering | University of Kufa</p>
        <p>📧 ali.alghanimi@uokufa.edu.iq | 🏢 EE Building, Room 301</p>
        <p style='font-size: 0.9rem; color: #666; margin-top: 1rem;'>
            © 2025 University of Kufa | All rights reserved
        </p>
    </div>
""", unsafe_allow_html=True)