# ═══════════════════════════════════════════════════════════════════
# WEEKS 4-6: INTEGRATED INTERACTIVE DEMONSTRATIONS
# Digital Control Systems | University of Kufa
# Dr. Ali Al-Ghanimi | Electrical Engineering Department
# 
# Week 4: Z-Transform & Inverse
# Week 5: Pulse Transfer Functions
# Week 6: Stability Analysis I (Jury Test & Routh Criterion)
# ═══════════════════════════════════════════════════════════════════

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import control as ct
import pandas as pd  # FIXED: Added missing import
from datetime import datetime
import sympy as sp  # For Week 6 bilinear transformation

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Weeks 4-6: Z-Transform, Pulse TF & Stability | Dr. Al-Ghanimi",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:ali.alghanimi@uokufa.edu.iq',
        'About': 'Digital Control Systems - University of Kufa'
    }
)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
if 'view_count' not in st.session_state:
    st.session_state.view_count = 0

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .demo-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .info-box {
        background: #d1ecf1;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    .danger-box {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
    <div class="main-header">
        <h1>🎓 Digital Control Systems</h1>
        <h2>Interactive Demonstrations: Weeks 4, 5 & 6</h2>
        <p style="font-size: 1.1rem; margin-top: 1rem;">
            <strong>Dr. Ali Al-Ghanimi</strong><br>
            Electrical Engineering Department<br>
            University of Kufa | Academic Year 2025
        </p>
    </div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR - WEEK SELECTION
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📚 Select Week")
    
    week_selection = st.selectbox(
        "Choose your week:",
        ["Week 4: Z-Transform & Inverse",
         "Week 5: Pulse Transfer Functions",
         "Week 6: Stability Analysis I"],
        key="week_selector"
    )
    
    st.markdown("---")
    
    # Demo selection based on week
    if "Week 4" in week_selection:
        st.markdown("### 📊 Week 4 Demos")
        demo_selection = st.radio(
            "Select Demo:",
            ["📊 Demo 1: Z-Transform Calculator",
             "🎯 Demo 2: ROC Visualizer", 
             "🔄 Demo 3: Inverse Methods",
             "📐 Demo 4: Partial Fractions",
             "🔧 Demo 5: Properties"],
            key="demo_nav_w4"
        )
    elif "Week 5" in week_selection:  # Week 5
        st.markdown("### 🎯 Week 5 Demos")
        demo_selection = st.radio(
            "Select Demo:",
            ["📊 Demo 1: PTF Calculator",
             "🎯 Demo 2: Block Diagram Analyzer", 
             "🔄 Demo 3: Difference Eq ↔ TF",
             "📐 Demo 4: Closed-Loop Systems",
             "🔧 Demo 5: Open/Closed Loop Comparison"],
            key="demo_nav_w5"
        )
    else:  # Week 6
        st.markdown("### 🎯 Week 6 Demos")
        demo_selection = st.radio(
            "Select Demo:",
            ["📊 Demo 1: Unit Circle & Stability",
             "🎯 Demo 2: Jury Test Step-by-Step", 
             "🔄 Demo 3: Parametric Stability (K Range)",
             "📐 Demo 4: Bilinear Transformation",
             "🔧 Demo 5: Routh Criterion in w-plane"],
            key="demo_nav_w6"
        )
    
    st.markdown("---")
    
    st.markdown("### 🎯 Learning Objectives")
    if "Week 4" in week_selection:
        with st.expander("Week 4 Objectives"):
            st.markdown("""
            1. ✅ Calculate z-transforms
            2. ✅ Determine ROC
            3. ✅ Apply inverse z-transform
            4. ✅ Master partial fractions
            5. ✅ Utilize transform properties
            """)
    elif "Week 5" in week_selection:
        with st.expander("Week 5 Objectives"):
            st.markdown("""
            1. ✅ Derive pulse transfer functions
            2. ✅ Analyze block diagrams
            3. ✅ Convert difference equations
            4. ✅ Calculate closed-loop TF
            5. ✅ Compare open vs closed loop
            """)
    else:  # Week 6
        with st.expander("Week 6 Objectives"):
            st.markdown("""
            1. ✅ Apply Jury stability test
            2. ✅ Construct Jury table systematically
            3. ✅ Use bilinear transformation
            4. ✅ Apply Routh-Hurwitz criterion
            5. ✅ Find stable parameter ranges
            6. ✅ Compare stability analysis methods
            """)
    
    st.markdown("---")
    
    st.markdown("### 📖 Resources")
    if "Week 4" in week_selection:
        st.info("""
        **Week 4:** lec_2.pdf, lec_4.pdf, Chakrabortty Ch.2
        """)
    elif "Week 5" in week_selection:
        st.info("""
        **Week 5:** DCS.pdf, notes_A2.pdf, Chakrabortty Ch.4
        """)
    else:  # Week 6
        st.info("""
        **Week 6:** Chakrabortty Ch.7 (pp. 230-244), DCS.pdf, DigitalControlTextBook Ch.4
        """)
    
    st.markdown("---")
    
    st.markdown("### 💬 Contact")
    st.markdown("""
    📧 ali.alghanimi@uokufa.edu.iq  
    🏢 EE Building, Room 301  
    🕐 Office Hours: Sun-Thu, 10-12 PM
    """)

# ═══════════════════════════════════════════════════════════════════
# WEEK 4 DEMOS
# ═══════════════════════════════════════════════════════════════════

if "Week 4" in week_selection:
    
    if "Demo 1" in demo_selection:
        st.markdown("## 📊 Demo 1: Z-Transform Calculator")
        st.markdown("Calculate z-transforms for standard discrete-time signals")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Controls")
            
            signal_type = st.selectbox(
                "Select Signal Type:",
                ["Unit Step", "Unit Ramp", "Exponential (a^k)", 
                 "Sine Wave", "Cosine Wave"]
            )
            
            if "Exponential" in signal_type:
                a_val = st.slider("Exponential base (a):", -1.5, 1.5, 0.5, 0.1)
            elif "Sine" in signal_type or "Cosine" in signal_type:
                freq = st.slider("Frequency (ω):", 0.1, 3.0, 1.0, 0.1)
                T = st.slider("Sampling period (T):", 0.01, 1.0, 0.1, 0.01)
            
            n_samples = st.slider("Number of samples:", 10, 50, 20, 5)
        
        with col2:
            st.markdown("### 📈 Visualization")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Time domain
            k = np.arange(0, n_samples)
            
            if signal_type == "Unit Step":
                x = np.ones_like(k, dtype=float)
                z_transform = "z/(z-1)"
                roc = "|z| > 1"
                poles = [1.0]
            elif signal_type == "Unit Ramp":
                x = k.astype(float)
                z_transform = "Tz/(z-1)²"
                roc = "|z| > 1"
                poles = [1.0, 1.0]
            elif signal_type == "Exponential (a^k)":
                x = a_val**k
                z_transform = f"z/(z-{a_val:.2f})"
                roc = f"|z| > {abs(a_val):.2f}"
                poles = [a_val]
            elif signal_type == "Sine Wave":
                x = np.sin(freq * k * T)
                z_transform = "Complex (see z-transform tables)"
                roc = "|z| > 1"
                poles = [np.exp(1j*freq*T), np.exp(-1j*freq*T)]
            else:  # Cosine
                x = np.cos(freq * k * T)
                z_transform = "Complex (see z-transform tables)"
                roc = "|z| > 1"
                poles = [np.exp(1j*freq*T), np.exp(-1j*freq*T)]
            
            # Plot time domain
            markerline, stemlines, baseline = ax1.stem(k, x, basefmt=' ')
            plt.setp(stemlines, 'linewidth', 2, 'color', 'blue')
            plt.setp(markerline, 'markersize', 8, 'color', 'blue')
            ax1.set_xlabel('Sample Index (k)', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Amplitude x[k]', fontsize=11, fontweight='bold')
            ax1.set_title(f'{signal_type}', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='k', linewidth=0.5)
            
            # Plot pole-zero map
            theta = np.linspace(0, 2*np.pi, 100)
            ax2.plot(np.cos(theta), np.sin(theta), 'b-', linewidth=2, label='Unit Circle')
            
            # Plot poles
            for pole in poles:
                if np.isreal(pole):
                    ax2.plot(pole.real, 0, 'rx', markersize=12, markeredgewidth=2.5)
                else:
                    ax2.plot(pole.real, pole.imag, 'rx', markersize=12, markeredgewidth=2.5)
                    if pole.imag != 0:  # Plot conjugate
                        ax2.plot(pole.real, -pole.imag, 'rx', markersize=12, markeredgewidth=2.5)
            
            ax2.axhline(y=0, color='k', linewidth=1)
            ax2.axvline(x=0, color='k', linewidth=1)
            ax2.set_xlabel('Real Part', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Imaginary Part', fontsize=11, fontweight='bold')
            ax2.set_title('Pole-Zero Map', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10)
            ax2.set_aspect('equal')
            ax2.set_xlim(-1.5, 1.5)
            ax2.set_ylim(-1.5, 1.5)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display formula
            st.markdown(f"""
            **Z-Transform:** $X(z) = {z_transform}$  
            **ROC:** ${roc}$
            
            **Source:** lec_2.pdf, Z-Transform tables
            """)
    
    elif "Demo 2" in demo_selection:
        st.markdown("## 🎯 Demo 2: Region of Convergence (ROC) Visualizer")
        st.markdown("Understand how ROC depends on signal type and pole locations")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Controls")
            
            signal_type = st.selectbox(
                "Signal Type:",
                ["Causal (right-sided)", "Anti-causal (left-sided)", "Two-sided"]
            )
            
            st.markdown("**Pole Location:**")
            pole_real = st.slider("Real part:", -1.5, 1.5, 0.5, 0.1)
            pole_imag = st.slider("Imaginary part:", -1.5, 1.5, 0.0, 0.1)
            
            pole = pole_real + 1j*pole_imag
            pole_mag = np.abs(pole)
        
        with col2:
            st.markdown("### 📊 ROC Visualization")
            
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Unit circle
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=2, label='Unit Circle', alpha=0.5)
            
            # Pole circle
            ax.plot(pole_mag*np.cos(theta), pole_mag*np.sin(theta), 'r-', linewidth=2.5, 
                    label=f'|z| = {pole_mag:.3f}')
            
            # Shade ROC
            if signal_type == "Causal (right-sided)":
                # ROC: |z| > |pole|
                for r in np.linspace(pole_mag, 2, 30):
                    ax.fill(r*np.cos(theta), r*np.sin(theta), 'green', alpha=0.008)
                ax.text(1.5, 1.5, f'ROC: |z| > {pole_mag:.3f}', fontsize=14, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            elif signal_type == "Anti-causal (left-sided)":
                # ROC: |z| < |pole|
                for r in np.linspace(0, pole_mag, 30):
                    ax.fill(r*np.cos(theta), r*np.sin(theta), 'blue', alpha=0.008)
                ax.text(-1.5, 1.5, f'ROC: |z| < {pole_mag:.3f}', fontsize=14, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            else:  # Two-sided
                # Two ROCs possible
                st.warning("Two-sided signals have multiple possible ROCs!")
            
            # Plot pole
            ax.plot(pole.real, pole.imag, 'rx', markersize=15, markeredgewidth=3, label='Pole')
            
            ax.axhline(y=0, color='k', linewidth=1)
            ax.axvline(x=0, color='k', linewidth=1)
            ax.set_xlabel('Real Part', fontsize=12, fontweight='bold')
            ax.set_ylabel('Imaginary Part', fontsize=12, fontweight='bold')
            ax.set_title('Region of Convergence in Z-Plane', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11, loc='lower left')
            ax.set_aspect('equal')
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.info("""
            **Key Concept:** ROC depends on signal type!
            - **Causal signals:** ROC is outside the outermost pole
            - **Anti-causal signals:** ROC is inside the innermost pole
            - **Two-sided signals:** ROC is a ring between poles
            
            **Source:** lec_2.pdf (Section on ROC)
            """)
    
    elif "Demo 3" in demo_selection:
        st.markdown("## 🔄 Demo 3: Inverse Z-Transform Methods")
        st.markdown("Compare long division vs partial fractions methods")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Transfer Function")
            
            st.markdown("$$X(z) = \\frac{z}{(z-p_1)(z-p_2)}$$")
            
            pole1 = st.slider("Pole 1 (p₁):", 0.1, 0.9, 0.5, 0.05)
            pole2 = st.slider("Pole 2 (p₂):", 0.1, 0.9, 0.3, 0.05)
            n_terms = st.slider("Number of terms:", 10, 30, 15, 5)
        
        with col2:
            st.markdown("### 📈 Results Comparison")
            
            # Create transfer function
            num = [1, 0]  # z
            den = [1, -(pole1+pole2), pole1*pole2]  # (z-p1)(z-p2)
            
            # Method 1: Using scipy (partial fractions internally)
            sys = signal.dlti(num, den, dt=1)
            t, y_pf = signal.dstep(sys, n=n_terms)
            y_pf = y_pf[0].flatten()
            
            # Method 2: Direct calculation using partial fractions
            # X(z) = A/(z-p1) + B/(z-p2)
            # Solving: A = 1/(p1-p2), B = -1/(p1-p2) [assuming numerator = 1]
            A = pole1 / (pole1 - pole2) if pole1 != pole2 else 0.5
            B = -pole2 / (pole1 - pole2) if pole1 != pole2 else 0.5
            
            k = np.arange(n_terms)
            y_analytical = A * pole1**k + B * pole2**k
            
            # Plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot 1: Partial Fractions Result
            markerline1, stemlines1, baseline1 = ax1.stem(k, y_pf, basefmt=' ')
            plt.setp(stemlines1, 'linewidth', 2, 'color', 'blue')
            plt.setp(markerline1, 'markersize', 8, 'color', 'blue')
            ax1.set_xlabel('Sample Index (k)', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Amplitude x[k]', fontsize=11, fontweight='bold')
            ax1.set_title('Partial Fractions Method', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='k', linewidth=0.5)
            
            # Plot 2: Analytical Result
            markerline2, stemlines2, baseline2 = ax2.stem(k, y_analytical, basefmt=' ')
            plt.setp(stemlines2, 'linewidth', 2, 'color', 'red')
            plt.setp(markerline2, 'markersize', 8, 'color', 'red')
            ax2.set_xlabel('Sample Index (k)', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Amplitude x[k]', fontsize=11, fontweight='bold')
            ax2.set_title('Analytical Solution', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='k', linewidth=0.5)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown(f"""
            **Partial Fraction Expansion:**
            
            $$X(z) = \\frac{{A}}{{z-{pole1}}} + \\frac{{B}}{{z-{pole2}}}$$
            
            where $A = {A:.4f}$ and $B = {B:.4f}$
            
            **Time-domain solution:**
            $$x[k] = {A:.4f} \\cdot ({pole1})^k + {B:.4f} \\cdot ({pole2})^k$$
            
            **Source:** lec_4.pdf (slide 17), Chakrabortty Ch.2
            """)
    
    elif "Demo 4" in demo_selection:
        st.markdown("## 📐 Demo 4: Partial Fractions Step-by-Step")
        st.markdown("Interactive partial fraction expansion with detailed steps")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ System Parameters")
            
            pole1 = st.slider("Pole 1 (p₁):", 0.1, 0.9, 0.5, 0.05, key="pf_p1")
            pole2 = st.slider("Pole 2 (p₂):", 0.1, 0.9, 0.3, 0.05, key="pf_p2")
            K = st.slider("Gain K:", 0.5, 2.0, 1.0, 0.1, key="pf_K")
            show_steps = st.checkbox("Show step-by-step solution", value=True)
        
        with col2:
            st.markdown("### 📊 Solution")
            
            if show_steps:
                st.markdown(f"""
                **Given Transfer Function:**
                $$X(z) = \\frac{{{K}z}}{{(z-{pole1})(z-{pole2})}}$$
                
                **Step 1:** Divide by z
                $$\\frac{{X(z)}}{{z}} = \\frac{{{K}}}{{(z-{pole1})(z-{pole2})}}$$
                
                **Step 2:** Partial Fraction Expansion
                $$\\frac{{X(z)}}{{z}} = \\frac{{A}}{{z-{pole1}}} + \\frac{{B}}{{z-{pole2}}}$$
                
                **Step 3:** Solve for A and B
                """)
                
                A = K * pole1 / (pole1 - pole2) if pole1 != pole2 else K/2
                B = -K * pole2 / (pole1 - pole2) if pole1 != pole2 else K/2
                
                st.markdown(f"""
                $$A = \\frac{{{K} \\cdot {pole1}}}{{{pole1} - {pole2}}} = {A:.4f}$$
                $$B = \\frac{{-{K} \\cdot {pole2}}}{{{pole1} - {pole2}}} = {B:.4f}$$
                
                **Step 4:** Multiply by z
                $$X(z) = \\frac{{{A:.4f}z}}{{z-{pole1}}} + \\frac{{{B:.4f}z}}{{z-{pole2}}}$$
                
                **Step 5:** Inverse z-transform
                $$x[k] = {A:.4f} \\cdot ({pole1})^k + {B:.4f} \\cdot ({pole2})^k$$
                """)
            
            # Plot result
            k = np.arange(0, 25)
            A = K * pole1 / (pole1 - pole2) if pole1 != pole2 else K/2
            B = -K * pole2 / (pole1 - pole2) if pole1 != pole2 else K/2
            x = A * pole1**k + B * pole2**k
            
            fig, ax = plt.subplots(figsize=(10, 5))
            markerline, stemlines, baseline = ax.stem(k, x, basefmt=' ')
            plt.setp(stemlines, 'linewidth', 2.5, 'color', 'purple')
            plt.setp(markerline, 'markersize', 10, 'color', 'purple')
            ax.set_xlabel('Sample Index (k)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Amplitude x[k]', fontsize=12, fontweight='bold')
            ax.set_title(f'Inverse Z-Transform Result (K={K})', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linewidth=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Stability check
            all_stable = (abs(pole1) < 1) and (abs(pole2) < 1)
            stability_text = "STABLE ✓" if all_stable else "UNSTABLE ✗"
            stability_color = "success" if all_stable else "warning"
            
            st.markdown(f"""
            <div class="{stability_color}-box">
                <h4>System Stability: {stability_text}</h4>
                <p>Pole 1: |{pole1}| = {abs(pole1):.3f} {'< 1 ✓' if abs(pole1) < 1 else '> 1 ✗'}</p>
                <p>Pole 2: |{pole2}| = {abs(pole2):.3f} {'< 1 ✓' if abs(pole2) < 1 else '> 1 ✗'}</p>
            </div>
            """, unsafe_allow_html=True)
    
    elif "Demo 5" in demo_selection:
        st.markdown("## 🔧 Demo 5: Z-Transform Properties")
        st.markdown("Visualize key z-transform properties interactively")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Select Property")
            
            property_type = st.selectbox(
                "Property:",
                ["Time Shifting (Delay)", "Time Shifting (Advance)", 
                 "Scaling (Multiplication by aⁿ)", "Linearity"]
            )
            
            if "Time Shifting" in property_type:
                k_shift = st.slider("Shift amount (k):", 1, 5, 2, 1)
            elif "Scaling" in property_type:
                a_scale = st.slider("Scaling factor (a):", 0.5, 1.5, 0.8, 0.1)
        
        with col2:
            st.markdown("### 📊 Visualization")
            
            n = np.arange(0, 20)
            x = 0.7**n  # Original signal
            
            if "Delay" in property_type:
                y = np.zeros_like(x)
                if k_shift < len(x):
                    y[k_shift:] = x[:-k_shift]
                property_name = f"Time Delay by {k_shift}"
                formula = f"y[n] = x[n-{k_shift}]"
                ztransform = f"Y(z) = z^{{-{k_shift}}} X(z)"
            elif "Advance" in property_type:
                y = np.zeros_like(x)
                if k_shift < len(x):
                    y[:-k_shift] = x[k_shift:]
                property_name = f"Time Advance by {k_shift}"
                formula = f"y[n] = x[n+{k_shift}]"
                ztransform = f"Y(z) = z^{{{k_shift}}} X(z)"
            elif "Scaling" in property_type:
                y = (a_scale**n) * x
                property_name = f"Scaling by a={a_scale}"
                formula = f"y[n] = {a_scale}^n · x[n]"
                ztransform = f"Y(z) = X(z/{a_scale})"
            else:  # Linearity
                x2 = 0.5**n
                y = 0.5*x + 0.3*x2
                property_name = "Linearity"
                formula = "y[n] = 0.5·x₁[n] + 0.3·x₂[n]"
                ztransform = "Y(z) = 0.5·X₁(z) + 0.3·X₂(z)"
            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
            
            # Original signal
            markerline1, stemlines1, baseline1 = ax1.stem(n, x, basefmt=' ')
            plt.setp(stemlines1, 'linewidth', 2, 'color', 'blue')
            plt.setp(markerline1, 'markersize', 8, 'color', 'blue')
            ax1.set_xlabel('n', fontsize=11, fontweight='bold')
            ax1.set_ylabel('x[n]', fontsize=11, fontweight='bold')
            ax1.set_title('Original Signal', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='k', linewidth=0.5)
            
            # Modified signal
            markerline2, stemlines2, baseline2 = ax2.stem(n, y, basefmt=' ')
            plt.setp(stemlines2, 'linewidth', 2, 'color', 'red')
            plt.setp(markerline2, 'markersize', 8, 'color', 'red')
            ax2.set_xlabel('n', fontsize=11, fontweight='bold')
            ax2.set_ylabel('y[n]', fontsize=11, fontweight='bold')
            ax2.set_title('Modified Signal', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='k', linewidth=0.5)
            
            # Overlay comparison
            ax3.stem(n, x, basefmt=' ', linefmt='b-', markerfmt='bo', label='Original')
            ax3.stem(n+0.2, y, basefmt=' ', linefmt='r--', markerfmt='ro', label='Modified')
            ax3.set_xlabel('n', fontsize=11, fontweight='bold')
            ax3.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
            ax3.set_title('Comparison', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.legend(fontsize=10)
            ax3.axhline(y=0, color='k', linewidth=0.5)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown(f"""
            **Property:** {property_name}
            
            **Time Domain:** {formula}
            
            **Z-Domain:** {ztransform}
            
            **Source:** lec_2.pdf (Z-Transform Properties Table)
            """)

# ═══════════════════════════════════════════════════════════════════
# WEEK 5 DEMOS
# ═══════════════════════════════════════════════════════════════════

elif "Week 5" in week_selection:
    
    if "Demo 1" in demo_selection:
        st.markdown("## 📊 Demo 1: Pulse Transfer Function (PTF) Calculator")
        st.markdown("Calculate PTF for systems with ZOH + continuous plant")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Plant Parameters")
            
            system_type = st.selectbox(
                "Plant Type:",
                ["First Order: 1/(s+a)", "Second Order: ω²/(s²+2ζωs+ω²)", 
                 "Integrator: 1/s", "Double Integrator: 1/s²"]
            )
            
            T = st.slider("Sampling Period T (sec):", 0.01, 1.0, 0.1, 0.01)
            
            if "First Order" in system_type:
                a = st.slider("Time constant (a):", 0.1, 10.0, 1.0, 0.1)
            elif "Second Order" in system_type:
                omega_n = st.slider("Natural frequency ωₙ:", 0.5, 10.0, 2.0, 0.5)
                zeta = st.slider("Damping ratio ζ:", 0.1, 1.5, 0.7, 0.1)
        
        with col2:
            st.markdown("### 📈 PTF Calculation")
            
            if "First Order" in system_type:
                # G(s) = 1/(s+a)
                # G(z) = (1-e^(-aT))z / (z - e^(-aT))
                Gs_num = [1]
                Gs_den = [1, a]
                
                st.latex(r"G(s) = \frac{1}{s+a}")
                
                # Calculate PTF
                sys_c = ct.TransferFunction(Gs_num, Gs_den)
                sys_d = ct.sample_system(sys_c, T, method='zoh')
                
                st.latex(f"G(z) = {sys_d}")
                
            elif "Second Order" in system_type:
                # G(s) = ω²/(s² + 2ζωs + ω²)
                Gs_num = [omega_n**2]
                Gs_den = [1, 2*zeta*omega_n, omega_n**2]
                
                st.latex(r"G(s) = \frac{\omega_n^2}{s^2 + 2\zeta\omega_n s + \omega_n^2}")
                
                sys_c = ct.TransferFunction(Gs_num, Gs_den)
                sys_d = ct.sample_system(sys_c, T, method='zoh')
                
                st.latex(f"G(z) = {sys_d}")
                
            elif "Integrator" in system_type:
                # G(s) = 1/s
                # G(z) = Tz/(z-1)
                Gs_num = [1]
                Gs_den = [1, 0]
                
                st.latex(r"G(s) = \frac{1}{s}")
                
                st.latex(rf"G(z) = \frac{{T \cdot z}}{{z-1}} = \frac{{{T}z}}{{z-1}}")
                
            else:  # Double Integrator
                # G(s) = 1/s²
                Gs_num = [1]
                Gs_den = [1, 0, 0]
                
                st.latex(r"G(s) = \frac{1}{s^2}")
                
                sys_c = ct.TransferFunction(Gs_num, Gs_den)
                sys_d = ct.sample_system(sys_c, T, method='zoh')
                
                st.latex(f"G(z) = {sys_d}")
            
            st.info("""
            **ZOH Transfer Function:**
            $$G_{ZOH}(s) = \\frac{1-e^{-Ts}}{s}$$
            
            **Complete PTF:**
            $$G(z) = (1-z^{-1}) \\mathcal{Z}\\left\\{\\frac{G(s)}{s}\\right\\}$$
            
            **Source:** DCS.pdf (Pulse Transfer Functions), lec_4.pdf
            """)
    
    elif "Demo 2" in demo_selection:
        st.markdown("## 🎯 Demo 2: Block Diagram Analyzer")
        st.markdown("Analyze different block diagram configurations")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Configuration")
            
            config_type = st.selectbox(
                "Block Diagram Type:",
                ["Cascaded (with sampler)", "Cascaded (no sampler)", 
                 "Feedback with sampler", "Feedback no sampler"]
            )
            
            G1_pole = st.slider("G₁ pole:", 0.1, 0.9, 0.5, 0.05)
            G2_pole = st.slider("G₂ pole:", 0.1, 0.9, 0.3, 0.05)
        
        with col2:
            st.markdown("### 📊 Analysis")
            
            if "Cascaded (with sampler)" in config_type:
                st.markdown("""
                **Configuration:** G₁(z) → Sampler → G₂(z)
                
                **Rule:** When systems are separated by a sampler:
                $$G(z) = G_1(z) \\cdot G_2(z)$$
                """)
                
                st.success("✓ Direct multiplication possible!")
                
            elif "Cascaded (no sampler)" in config_type:
                st.markdown("""
                **Configuration:** G₁(s) → G₂(s) → Sampler
                
                **Rule:** When NO sampler between systems:
                $$G(z) = \\mathcal{Z}\\{G_1(s) \\cdot G_2(s)\\}$$
                
                **NOT equal to:** $G_1(z) \\cdot G_2(z)$
                """)
                
                st.warning("⚠️ Must take z-transform of PRODUCT G₁(s)G₂(s)!")
                
            elif "Feedback with sampler" in config_type:
                st.markdown("""
                **Configuration:** Feedback with sampler in forward path
                
                **Closed-loop TF:**
                $$\\frac{C(z)}{R(z)} = \\frac{G(z)}{1 + G(z)H(z)}$$
                """)
                
                # Example calculation
                G_z = f"z/(z-{G1_pole})"
                H_z = "1"
                
                st.latex(f"G(z) = {G_z}, \\ H(z) = {H_z}")
                st.latex(rf"\frac{{C(z)}}{{R(z)}} = \frac{{z/(z-{G1_pole})}}{{1 + z/(z-{G1_pole})}}")
                
            else:  # Feedback no sampler
                st.markdown("""
                **Configuration:** Feedback with sampler in feedback path
                
                **Important:** Input R(s) is NOT sampled!
                
                Cannot define simple R(z) → C(z) transfer function.
                Must work with starred transform C*(s).
                """)
                
                st.warning("⚠️ More complex analysis required!")
            
            st.info("""
            **Key Rules from DCS.pdf:**
            1. Systems separated by sampler → multiply z-transforms
            2. Systems NOT separated → z-transform of product
            3. Feedback loops → use standard feedback formula
            
            **Source:** DCS.pdf (Block Diagrams), Chakrabortty Ch.4
            """)
    
    elif "Demo 3" in demo_selection:
        st.markdown("## 🔄 Demo 3: Difference Equation ↔ Transfer Function")
        st.markdown("Convert between time-domain and z-domain representations")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ System Parameters")
            
            conversion_dir = st.radio(
                "Conversion Direction:",
                ["Difference Eq → TF", "TF → Difference Eq"]
            )
            
            if "Difference Eq" in conversion_dir:
                st.markdown("**Difference Equation:**")
                st.markdown("$y[k] + a_1 y[k-1] + a_2 y[k-2] = b_0 u[k] + b_1 u[k-1]$")
                
                a1 = st.slider("Coefficient a₁:", -2.0, 2.0, -1.5, 0.1)
                a2 = st.slider("Coefficient a₂:", -1.0, 1.0, 0.5, 0.1)
                b0 = st.slider("Coefficient b₀:", 0.0, 2.0, 1.0, 0.1)
                b1 = st.slider("Coefficient b₁:", 0.0, 2.0, 0.5, 0.1)
            else:
                st.markdown("**Transfer Function:**")
                st.markdown("$G(z) = \\frac{b_0 + b_1 z^{-1}}{1 + a_1 z^{-1} + a_2 z^{-2}}$")
                
                a1 = st.slider("Coefficient a₁:", -2.0, 2.0, -1.5, 0.1, key="tf_a1")
                a2 = st.slider("Coefficient a₂:", -1.0, 1.0, 0.5, 0.1, key="tf_a2")
                b0 = st.slider("Coefficient b₀:", 0.0, 2.0, 1.0, 0.1, key="tf_b0")
                b1 = st.slider("Coefficient b₁:", 0.0, 2.0, 0.5, 0.1, key="tf_b1")
        
        with col2:
            st.markdown("### 📊 Conversion Result")
            
            if "Difference Eq" in conversion_dir:
                st.markdown("""
                **Given Difference Equation:**
                """)
                st.latex(f"y[k] + {a1}y[k-1] + {a2}y[k-2] = {b0}u[k] + {b1}u[k-1]")
                
                st.markdown("""
                **Step 1:** Take z-transform (assuming zero initial conditions)
                """)
                st.latex(f"Y(z) + {a1}z^{{-1}}Y(z) + {a2}z^{{-2}}Y(z) = {b0}U(z) + {b1}z^{{-1}}U(z)")
                
                st.markdown("""
                **Step 2:** Solve for G(z) = Y(z)/U(z)
                """)
                st.latex(f"G(z) = \\frac{{{b0} + {b1}z^{{-1}}}}{{1 + {a1}z^{{-1}} + {a2}z^{{-2}}}}")
                
                st.markdown("""
                **Step 3:** Multiply by z² to get standard form
                """)
                st.latex(f"G(z) = \\frac{{{b0}z^2 + {b1}z}}{{z^2 + {a1}z + {a2}}}")
                
            else:  # TF to Diff Eq
                st.markdown("""
                **Given Transfer Function:**
                """)
                st.latex(f"G(z) = \\frac{{{b0} + {b1}z^{{-1}}}}{{1 + {a1}z^{{-1}} + {a2}z^{{-2}}}}")
                
                st.markdown("""
                **Step 1:** Cross-multiply
                """)
                st.latex(f"Y(z)(1 + {a1}z^{{-1}} + {a2}z^{{-2}}) = U(z)({b0} + {b1}z^{{-1}})")
                
                st.markdown("""
                **Step 2:** Take inverse z-transform
                """)
                st.latex(f"y[k] + {a1}y[k-1] + {a2}y[k-2] = {b0}u[k] + {b1}u[k-1]")
                
                st.markdown("""
                **Step 3:** Solve for y[k] (recursive form)
                """)
                st.latex(f"y[k] = {b0}u[k] + {b1}u[k-1] - {a1}y[k-1] - {a2}y[k-2]")
            
            # Calculate and plot poles
            poles = np.roots([1, a1, a2])
            
            fig, ax = plt.subplots(figsize=(8, 8))
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta), 'b-', linewidth=2, label='Unit Circle')
            
            for pole in poles:
                ax.plot(pole.real, pole.imag, 'rx', markersize=15, markeredgewidth=3)
            
            ax.axhline(y=0, color='k', linewidth=1)
            ax.axvline(x=0, color='k', linewidth=1)
            ax.set_xlabel('Real Part', fontsize=12, fontweight='bold')
            ax.set_ylabel('Imaginary Part', fontsize=12, fontweight='bold')
            ax.set_title('Pole Locations', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11)
            ax.set_aspect('equal')
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            all_stable = all(np.abs(poles) < 1)
            st.markdown(f"""
            **Stability:** {'STABLE ✓' if all_stable else 'UNSTABLE ✗'}
            
            **Source:** DCS.pdf (Difference Equations), notes_A2.pdf
            """)
    
    elif "Demo 4" in demo_selection:
        st.markdown("## 📐 Demo 4: Closed-Loop System Analysis")
        st.markdown("Analyze closed-loop digital control systems")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ System Parameters")
            
            pole1 = st.slider("Open-loop pole 1:", 0.1, 0.9, 0.5, 0.05, key="cl_p1")
            pole2 = st.slider("Open-loop pole 2:", 0.1, 0.9, 0.3, 0.05, key="cl_p2")
            K = st.slider("Controller gain K:", 0.1, 5.0, 1.0, 0.1, key="cl_K")
            
            st.markdown(f"""
            **Open-Loop Transfer Function:**
            $$G(z) = \\frac{{K}}{{(z-{pole1})(z-{pole2})}}$$
            
            **Feedback:** Unity (H(z) = 1)
            """)
        
        with col2:
            st.markdown("### 📊 Closed-Loop Analysis")
            
            # Calculate closed-loop poles
            # Characteristic eq: (z-pole1)(z-pole2) + K = 0
            # z² - (pole1+pole2)z + pole1*pole2 + K = 0
            
            a = 1
            b = -(pole1 + pole2)
            c = pole1*pole2 + K
            
            cl_poles = np.roots([a, b, c])
            
            fig, ax = plt.subplots(figsize=(9, 9))
            
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta), 'b-', linewidth=2.5, label='Unit Circle')
            
            # Shade stable region
            for r in np.linspace(0, 1, 25):
                ax.fill(r*np.cos(theta), r*np.sin(theta), 'green', alpha=0.01)
            
            # Open-loop poles
            ax.plot([pole1, pole2], [0, 0], 'bx', markersize=12, markeredgewidth=2.5, 
                    label='Open-Loop Poles')
            
            # Closed-loop poles
            ax.plot(cl_poles.real, cl_poles.imag, 'ro', markersize=12, markeredgewidth=2.5,
                    label=f'Closed-Loop Poles (K={K})')
            
            ax.axhline(y=0, color='k', linewidth=1)
            ax.axvline(x=0, color='k', linewidth=1)
            ax.set_xlabel('Real Part', fontsize=12, fontweight='bold')
            ax.set_ylabel('Imaginary Part', fontsize=12, fontweight='bold')
            ax.set_title(f'Pole Movement with Gain K={K}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            ax.set_aspect('equal')
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_facecolor('#f8f9fa')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Analysis
            all_stable = all(np.abs(cl_poles) < 1)
            
            st.markdown(f"""
            **Characteristic Equation:**  
            $$1 + G(z)H(z) = 0$$  
            $$z^2 - {pole1+pole2:.2f}z + {pole1*pole2+K:.2f} = 0$$
            """)
            
            st.markdown(f"""
            <div class="{'success' if all_stable else 'warning'}-box">
                <h4>{'✅' if all_stable else '⚠️'} Closed-Loop Stability</h4>
                <p><strong>Closed-Loop Poles:</strong></p>
                <ul>
                    <li>z₁ = {cl_poles[0]:.4f} (|z₁| = {abs(cl_poles[0]):.4f})</li>
                    <li>z₂ = {cl_poles[1]:.4f} (|z₂| = {abs(cl_poles[1]):.4f})</li>
                </ul>
                <p><strong>Result:</strong> System is <strong>{'STABLE' if all_stable else 'UNSTABLE'}</strong></p>
                <p>{'All poles inside unit circle ✓' if all_stable else 'Poles outside unit circle - reduce gain K!'}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("""
            **Next Week:** Learn Jury Test to find stable K range algebraically!
            
            **Source:** DCS.pdf (Closed-Loop Systems), Chakrabortty Ch.4
            """)
    
    elif "Demo 5" in demo_selection:
        st.markdown("## 🔧 Demo 5: Open-Loop vs Closed-Loop Comparison")
        st.markdown("Compare open-loop and closed-loop system responses")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ System Setup")
            
            plant_pole = st.slider("Plant pole:", 0.1, 0.9, 0.5, 0.05)
            K_ol = st.slider("Open-loop gain:", 0.5, 3.0, 1.0, 0.1, key="comp_ol")
            K_cl = st.slider("Closed-loop gain:", 0.5, 3.0, 1.5, 0.1, key="comp_cl")
            
            n_steps = 30
        
        with col2:
            st.markdown("### 📊 Step Response Comparison")
            
            # Open-loop system
            num_ol = [K_ol]
            den_ol = [1, -plant_pole]
            sys_ol = signal.dlti(num_ol, den_ol, dt=1)
            t_ol, y_ol = signal.dstep(sys_ol, n=n_steps)
            y_ol = y_ol[0].flatten()
            
            # Closed-loop system
            # G_cl(z) = K/(z-pole+K)
            num_cl = [K_cl]
            den_cl = [1, -(plant_pole-K_cl)]
            sys_cl = signal.dlti(num_cl, den_cl, dt=1)
            t_cl, y_cl = signal.dstep(sys_cl, n=n_steps)
            y_cl = y_cl[0].flatten()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            k = np.arange(n_steps)
            ax.step(k, y_ol, 'b-', linewidth=2.5, where='post', label='Open-Loop')
            ax.step(k, y_cl, 'r-', linewidth=2.5, where='post', label='Closed-Loop')
            ax.axhline(y=1, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Reference')
            
            ax.set_xlabel('Sample Index (k)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Output y[k]', fontsize=12, fontweight='bold')
            ax.set_title('Step Response: Open-Loop vs Closed-Loop', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11, loc='best')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Calculate steady-state errors
            ss_error_ol = abs(1 - y_ol[-1])
            ss_error_cl = abs(1 - y_cl[-1])
            
            st.markdown("""
            **Comparison Table:**
            """)
            
            comparison_data = {
                "Characteristic": ["Steady-State Error", "Disturbance Rejection", 
                                  "Sensitivity", "Complexity"],
                "Open-Loop": [f"{ss_error_ol:.3f}", "Poor", "High", "Simple"],
                "Closed-Loop": [f"{ss_error_cl:.3f}", "Good", "Low", "Complex"]
            }
            
            df = pd.DataFrame(comparison_data)
            st.table(df)
            
            st.markdown(f"""
            <div class="info-box">
                <h4>📊 Key Observations</h4>
                <ul>
                    <li><strong>Open-Loop:</strong> Simple but sensitive to disturbances</li>
                    <li><strong>Closed-Loop:</strong> Better performance, requires more design</li>
                    <li><strong>Trade-off:</strong> Complexity vs robustness</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.success("""
            **Why Feedback?**
            - Reduces steady-state error
            - Improves disturbance rejection
            - Reduces parameter sensitivity
            - Enables control of unstable plants
            
            **Source:** DCS.pdf (Feedback Benefits), Chakrabortty Ch.1
            """)

# ═══════════════════════════════════════════════════════════════════
# WEEK 6: HELPER FUNCTIONS FOR STABILITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def draw_unit_circle(ax, shade_stable=True):
    """Draw unit circle with optional shading of stable region"""
    theta = np.linspace(0, 2*np.pi, 200)
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)
    
    ax.plot(x_circle, y_circle, 'b-', linewidth=2.5, label='Unit Circle (|z|=1)')
    
    if shade_stable:
        for r in np.linspace(0, 1, 30):
            ax.fill(r*x_circle, r*y_circle, 'green', alpha=0.01)
        ax.text(0, 0, 'STABLE\nREGION', ha='center', va='center', 
                fontsize=14, fontweight='bold', color='darkgreen', alpha=0.6)
    
    ax.axhline(y=0, color='k', linewidth=0.8, alpha=0.5)
    ax.axvline(x=0, color='k', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Real Part', fontsize=12, fontweight='bold')
    ax.set_ylabel('Imaginary Part', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    return ax

def jury_test(coeffs):
    """
    Perform Jury stability test
    Source: Chakrabortty Ch.7, Section 7.5, pp. 239-242
           DigitalControlTextBook.pdf, Chapter 4.5, pp. 104-109
    """
    n = len(coeffs) - 1
    a = np.array(coeffs)
    
    if a[-1] <= 0:
        a = -a
    
    conditions = []
    table = [a.copy()]
    
    # Condition 1: Q(1) > 0
    Q_1 = np.sum(a)
    cond1 = Q_1 > 0
    conditions.append(("Q(1) > 0", Q_1, cond1))
    
    # Condition 2: (-1)^n * Q(-1) > 0
    Q_minus1 = sum([a[i] * ((-1)**i) for i in range(len(a))])
    Q_minus1_test = ((-1)**n) * Q_minus1
    cond2 = Q_minus1_test > 0
    conditions.append(("(-1)^n Q(-1) > 0", Q_minus1_test, cond2))
    
    # Condition 3: |a0| < an
    cond3 = abs(a[0]) < a[-1]
    conditions.append(("|a0| < an", f"|{a[0]:.4f}| < {a[-1]:.4f}", cond3))
    
    if not all([cond1, cond2, cond3]):
        return False, conditions, table
    
    # Build Jury table
    current_row = a.copy()
    row_num = 3
    
    while len(current_row) > 2:
        table.append(current_row[::-1])
        
        next_row = []
        for k in range(len(current_row) - 1):
            element = current_row[0] * current_row[k] - current_row[-1] * current_row[-(k+1)]
            next_row.append(element)
        
        table.append(np.array(next_row))
        
        if len(next_row) > 1:
            cond = abs(next_row[0]) > abs(next_row[-1])
            conditions.append((f"|b{row_num-2}_0| > |b{row_num-2}_{len(next_row)-1}|", 
                             f"|{next_row[0]:.4f}| > |{next_row[-1]:.4f}|", cond))
            if not cond:
                return False, conditions, table
        
        current_row = np.array(next_row)
        row_num += 2
    
    return all([c[2] for c in conditions]), conditions, table

def bilinear_transform(z_poly):
    """
    Apply bilinear transformation w = (z+1)/(z-1)
    Source: Chakrabortty Ch.7, Section 7.3, pp. 234-236
    """
    z = sp.Symbol('z')
    w = sp.Symbol('w')
    
    poly_z = sum([z_poly[i] * z**i for i in range(len(z_poly))])
    poly_w = poly_z.subs(z, (w+1)/(w-1))
    
    n = len(z_poly) - 1
    poly_w = sp.simplify(poly_w * (w-1)**n)
    
    poly_w = sp.expand(poly_w)
    w_coeffs = [float(poly_w.coeff(w, i)) for i in range(sp.degree(poly_w, w) + 1)]
    
    return w_coeffs, poly_w

def routh_array(coeffs):
    """
    Construct Routh array
    Source: Standard control theory (prerequisite)
    """
    coeffs = coeffs[::-1]
    n = len(coeffs) - 1
    
    routh = np.zeros((n + 1, (n + 2) // 2))
    
    routh[0, :] = coeffs[0::2]
    routh[1, :len(coeffs[1::2])] = coeffs[1::2]
    
    for i in range(2, n + 1):
        for j in range((n + 2) // 2 - 1):
            if routh[i-1, 0] == 0:
                routh[i-1, 0] = 1e-10
            
            if j < routh.shape[1] - 1:
                routh[i, j] = (routh[i-1, 0] * routh[i-2, j+1] - 
                              routh[i-2, 0] * routh[i-1, j+1]) / routh[i-1, 0]
    
    first_col = routh[:, 0]
    first_col = first_col[first_col != 0]
    
    sign_changes = 0
    for i in range(len(first_col) - 1):
        if first_col[i] * first_col[i+1] < 0:
            sign_changes += 1
    
    stable = sign_changes == 0
    
    return routh, stable, first_col

# ═══════════════════════════════════════════════════════════════════
# WEEK 6 DEMOS
# ═══════════════════════════════════════════════════════════════════

if "Week 6" in week_selection:
    
    if "Demo 1" in demo_selection:
        st.markdown("## 📊 Demo 1: Unit Circle & Stability Regions")
        st.markdown("Visualize pole locations and determine stability in the z-plane")
        
        st.markdown("""
        <div class="info-box">
            <h4>📚 Theory (Source: Chakrabortty Ch.7, pp. 230-232)</h4>
            <p>For discrete-time systems, stability is determined by pole locations:</p>
            <ul>
                <li><strong>STABLE:</strong> All poles inside unit circle (|z| < 1)</li>
                <li><strong>MARGINALLY STABLE:</strong> Simple poles on unit circle (|z| = 1)</li>
                <li><strong>UNSTABLE:</strong> Any pole outside unit circle (|z| > 1)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Add Poles")
            
            st.markdown("**Pole 1:**")
            p1_real = st.slider("Real part:", -1.5, 1.5, 0.5, 0.05, key="p1r")
            p1_imag = st.slider("Imaginary part:", -1.5, 1.5, 0.3, 0.05, key="p1i")
            
            st.markdown("**Pole 2:**")
            p2_real = st.slider("Real part:", -1.5, 1.5, -0.4, 0.05, key="p2r")
            p2_imag = st.slider("Imaginary part:", -1.5, 1.5, 0.2, 0.05, key="p2i")
            
            st.markdown("**Pole 3:**")
            p3_real = st.slider("Real part:", -1.5, 1.5, 0.2, 0.05, key="p3r")
            p3_imag = st.slider("Imaginary part:", -1.5, 1.5, -0.6, 0.05, key="p3i")
            
            poles = [
                complex(p1_real, p1_imag),
                complex(p2_real, p2_imag),
                complex(p3_real, p3_imag)
            ]
        
        with col2:
            st.markdown("### 📈 Z-Plane Visualization")
            
            fig, ax = plt.subplots(figsize=(10, 10))
            draw_unit_circle(ax, shade_stable=True)
            
            for i, pole in enumerate(poles):
                mag = abs(pole)
                if mag < 1:
                    color, marker, label_suffix = 'green', 'o', 'STABLE'
                elif abs(mag - 1) < 0.05:
                    color, marker, label_suffix = 'orange', 's', 'MARGINAL'
                else:
                    color, marker, label_suffix = 'red', 'x', 'UNSTABLE'
                
                ax.plot(pole.real, pole.imag, marker, color=color, markersize=15, 
                       markeredgewidth=3, label=f'Pole {i+1}: {label_suffix}')
                
                ax.plot([0, pole.real], [0, pole.imag], 'k--', alpha=0.3, linewidth=1)
                
                ax.text(pole.real, pole.imag + 0.15, f'|z|={mag:.3f}', 
                       ha='center', fontsize=9, fontweight='bold')
            
            ax.set_title('Pole Locations in Z-Plane', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            all_stable = all(abs(p) < 1 for p in poles)
            any_marginal = any(0.95 < abs(p) < 1.05 for p in poles)
            any_unstable = any(abs(p) > 1 for p in poles)
            
            if all_stable:
                st.markdown("""
                <div class="success-box">
                    <h4>✅ System is STABLE</h4>
                    <p>All poles are strictly inside the unit circle.</p>
                </div>
                """, unsafe_allow_html=True)
            elif any_unstable:
                st.markdown("""
                <div class="danger-box">
                    <h4>❌ System is UNSTABLE</h4>
                    <p>At least one pole is outside the unit circle.</p>
                </div>
                """, unsafe_allow_html=True)
            elif any_marginal:
                st.markdown("""
                <div class="warning-box">
                    <h4>⚠️ System is MARGINALLY STABLE</h4>
                    <p>Poles on the unit circle cause sustained oscillations.</p>
                </div>
                """, unsafe_allow_html=True)
            
            pole_data = {
                "Pole": ["z₁", "z₂", "z₃"],
                "Value": [f"{p.real:.3f} + {p.imag:.3f}j" for p in poles],
                "Magnitude": [f"{abs(p):.4f}" for p in poles],
                "Status": ["✅ Stable" if abs(p) < 1 else 
                          "⚠️ Marginal" if 0.95 < abs(p) < 1.05 else 
                          "❌ Unstable" for p in poles]
            }
            df = pd.DataFrame(pole_data)
            st.table(df)
    
    elif "Demo 2" in demo_selection:
        st.markdown("## 🎯 Demo 2: Jury Test - Step by Step")
        st.markdown("Apply Jury stability test with detailed calculations")
        
        st.markdown("""
        <div class="info-box">
            <h4>📚 Jury Test (Source: Chakrabortty Ch.7, pp. 239-242)</h4>
            <p>For Q(z) = aₙzⁿ + ... + a₁z + a₀ = 0:</p>
            <p><strong>(n+1) Conditions:</strong></p>
            <ol>
                <li>Q(1) > 0</li>
                <li>(-1)ⁿ Q(-1) > 0</li>
                <li>|a₀| < aₙ</li>
                <li>Jury table conditions</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Polynomial Coefficients")
            
            order = st.selectbox("Order:", [2, 3, 4], index=1)
            
            st.markdown(f"**Q(z) = aₙzⁿ + ... + a₀:**")
            
            coeffs = []
            for i in range(order + 1):
                val = st.number_input(f"a_{i}:", value=0.0 if i < order else 1.0, 
                                     step=0.1, format="%.4f", key=f"coeff_{i}")
                coeffs.append(val)
            
            st.markdown("**Examples:**")
            if st.button("Example 1: Stable"):
                coeffs = [0.08, -0.6, 1.0]
            if st.button("Example 2: Unstable"):
                coeffs = [0.3, 0.07, -1.2, 1.0]
        
        with col2:
            st.markdown("### 📊 Jury Test Execution")
            
            if st.button("🚀 Run Jury Test", type="primary"):
                
                stable, conditions, table = jury_test(coeffs)
                
                st.markdown("#### Step-by-Step Analysis")
                
                for idx, (cond_name, cond_value, cond_result) in enumerate(conditions):
                    if cond_result:
                        st.success(f"✅ Condition {idx+1}: {cond_name} → {cond_value} ✓")
                    else:
                        st.error(f"❌ Condition {idx+1}: {cond_name} → {cond_value} ✗")
                        if idx < 3:
                            st.stop()
                
                st.markdown("#### Jury Table")
                
                table_display = []
                for i, row in enumerate(table):
                    row_label = f"Row {i+1}"
                    row_data = [f"{x:.6f}" for x in row]
                    table_display.append([row_label] + row_data)
                
                max_len = max(len(row) for row in table)
                table_df = pd.DataFrame(table_display, 
                                       columns=["Row"] + [f"Col {j}" for j in range(max_len)])
                st.dataframe(table_df, use_container_width=True)
                
                st.markdown("---")
                if stable:
                    st.markdown("""
                    <div class="success-box">
                        <h3>✅ SYSTEM IS STABLE</h3>
                        <p>All Jury conditions satisfied!</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="danger-box">
                        <h3>❌ SYSTEM IS UNSTABLE</h3>
                        <p>At least one condition failed.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("#### Verification")
                roots = np.roots(coeffs[::-1])
                
                fig, ax = plt.subplots(figsize=(8, 8))
                draw_unit_circle(ax)
                
                for i, root in enumerate(roots):
                    mag = abs(root)
                    color = 'green' if mag < 1 else 'red' if mag > 1 else 'orange'
                    ax.plot(root.real, root.imag, 'o', color=color, markersize=12,
                           label=f'z_{i+1} = {root:.4f} (|z|={mag:.4f})')
                
                ax.set_title('Computed Poles', fontsize=13, fontweight='bold')
                ax.legend(loc='best', fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
    
    elif "Demo 3" in demo_selection:
        st.markdown("## 🔄 Demo 3: Parametric Stability (Find K Range)")
        st.markdown("Determine stable range of gain K")
        
        st.markdown("""
        <div class="info-box">
            <h4>📚 Parametric Analysis (Source: Chakrabortty Ex 7.4, p.242)</h4>
            <p>Apply Jury conditions to find K constraints</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ System")
            
            st.markdown("**Q(z) = z² + (aK + b)z + (c + dK) = 0**")
            
            a = st.number_input("Coefficient 'a':", value=0.368, format="%.4f")
            b = st.number_input("Constant 'b':", value=-1.368, format="%.4f")
            c = st.number_input("Constant 'c':", value=0.368, format="%.4f")
            d = st.number_input("Coefficient 'd':", value=0.264, format="%.4f")
            
            K_test = st.slider("Test gain K:", 0.0, 5.0, 1.2, 0.1)
        
        with col2:
            st.markdown("### 📈 Stability Analysis")
            
            st.markdown("#### Condition 1: Q(1) > 0")
            Q_1_coeff = a + d
            Q_1_const = 1 + b + c
            
            if Q_1_coeff > 0:
                K_lower = -Q_1_const / Q_1_coeff
                st.success(f"K > {K_lower:.4f}")
            
            st.markdown("#### Condition 2: Q(-1) > 0")
            Q_minus1_coeff = -a + d
            Q_minus1_const = 1 - b + c
            
            if Q_minus1_coeff != 0:
                K_upper_2 = -Q_minus1_const / Q_minus1_coeff if Q_minus1_coeff < 0 else float('inf')
                if K_upper_2 < float('inf'):
                    st.success(f"K < {K_upper_2:.4f}")
            
            st.markdown("#### Condition 3: |a₀| < 1")
            if d > 0:
                K_upper_3 = (1 - c) / d
                st.success(f"K < {K_upper_3:.4f}")
            
            st.markdown("---")
            st.markdown("### 🎯 Stable Range")
            
            K_min = max(0, K_lower)
            K_max = min(K_upper_2, K_upper_3) if K_upper_2 < float('inf') else K_upper_3
            
            if K_max > K_min:
                st.markdown(f"""
                <div class="success-box">
                    <h3>Stable Range:</h3>
                    <h2 style="text-align:center; color:green;">{K_min:.4f} < K < {K_max:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                K_range = np.linspace(0, K_max * 1.5, 200)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                ax.fill_between(K_range, 0, 1, 
                               where=[k < K_min or k > K_max for k in K_range], 
                               color='red', alpha=0.3, label='Unstable')
                ax.fill_between([K_min, K_max], 0, 1, 
                               color='green', alpha=0.3, label='Stable')
                
                ax.axvline(K_min, color='blue', linestyle='--', linewidth=2, label=f'K_min={K_min:.3f}')
                ax.axvline(K_max, color='red', linestyle='--', linewidth=2, label=f'K_max={K_max:.3f}')
                
                if K_min <= K_test <= K_max:
                    ax.axvline(K_test, color='green', linestyle='-', linewidth=2.5, 
                              label=f'K={K_test:.2f} (STABLE)')
                else:
                    ax.axvline(K_test, color='darkred', linestyle='-', linewidth=2.5, 
                              label=f'K={K_test:.2f} (UNSTABLE)')
                
                ax.set_xlabel('Gain K', fontsize=13, fontweight='bold')
                ax.set_title('Stability Region', fontsize=14, fontweight='bold')
                ax.set_ylim(0, 1)
                ax.set_yticks([])
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3, axis='x')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                if K_min <= K_test <= K_max:
                    st.success(f"✅ K = {K_test} is STABLE")
                else:
                    st.error(f"❌ K = {K_test} is UNSTABLE")
    
    elif "Demo 4" in demo_selection:
        st.markdown("## 📐 Demo 4: Bilinear Transformation")
        st.markdown("Transform z-plane to w-plane")
        
        st.markdown("""
        <div class="info-box">
            <h4>📚 Bilinear Transform (Source: Chakrabortty Ch.7, pp.234-236)</h4>
            <p><strong>w = (z+1)/(z-1)</strong></p>
            <p>Maps: Inside unit circle ↔ Left-half plane</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Polynomial in z")
            
            order = st.selectbox("Order:", [2, 3], index=0, key="bt_order")
            
            coeffs_z = []
            for i in range(order + 1):
                val = st.number_input(f"a_{i}:", value=1.0 if i == order else 
                                     (-0.6 if i == 1 else 0.08), 
                                     format="%.4f", key=f"bt_coeff_{i}")
                coeffs_z.append(val)
        
        with col2:
            st.markdown("### 🔄 Transformation")
            
            if st.button("🚀 Apply Transform", type="primary"):
                
                try:
                    w_coeffs, poly_w = bilinear_transform(coeffs_z)
                    
                    st.success("✅ Transformation successful!")
                    st.latex(f"Q(w) = {sp.latex(poly_w)}")
                    
                    roots_z = np.roots(coeffs_z[::-1])
                    roots_w = np.roots(w_coeffs[::-1])
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                    
                    # Z-plane
                    draw_unit_circle(ax1)
                    for root in roots_z:
                        mag = abs(root)
                        color = 'green' if mag < 1 else 'red'
                        ax1.plot(root.real, root.imag, 'o', color=color, markersize=12)
                    ax1.set_title('Z-Plane', fontsize=13, fontweight='bold')
                    
                    # W-plane
                    ax2.axvline(x=0, color='b', linewidth=2.5, label='Stability Boundary')
                    ax2.fill_betweenx([-2, 2], -2, 0, color='green', alpha=0.2, label='Stable (LHP)')
                    ax2.fill_betweenx([-2, 2], 0, 2, color='red', alpha=0.2, label='Unstable (RHP)')
                    
                    for root in roots_w:
                        color = 'green' if root.real < 0 else 'red'
                        ax2.plot(root.real, root.imag, 'o', color=color, markersize=12)
                    
                    ax2.set_xlabel('Real (σ)', fontsize=12, fontweight='bold')
                    ax2.set_ylabel('Imaginary (jω)', fontsize=12, fontweight='bold')
                    ax2.set_title('W-Plane', fontsize=13, fontweight='bold')
                    ax2.grid(True, alpha=0.3)
                    ax2.legend(fontsize=9)
                    ax2.set_xlim(-2, 2)
                    ax2.set_ylim(-2, 2)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    elif "Demo 5" in demo_selection:
        st.markdown("## 🔧 Demo 5: Routh Criterion in w-plane")
        st.markdown("Apply Routh-Hurwitz after bilinear transformation")
        
        st.markdown("""
        <div class="info-box">
            <h4>📚 Routh Criterion (Source: Chakrabortty Ch.7, pp.236-239)</h4>
            <p><strong>Procedure:</strong></p>
            <ol>
                <li>Transform Q(z) → Q(w)</li>
                <li>Construct Routh array</li>
                <li>Check first column signs</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Input")
            
            method = st.radio("Method:", ["From z-polynomial", "Direct w-polynomial"])
            
            if method == "From z-polynomial":
                order = st.selectbox("Order:", [2, 3], index=0, key="r_order")
                
                coeffs_z = []
                for i in range(order + 1):
                    val = st.number_input(f"a_{i}:", value=1.0 if i == order else 
                                         (0.2 if i == 1 else -0.5), 
                                         format="%.4f", key=f"r_z_{i}")
                    coeffs_z.append(val)
                
                if st.button("Transform"):
                    w_coeffs, _ = bilinear_transform(coeffs_z)
                    st.session_state.w_coeffs = w_coeffs
            else:
                order = st.selectbox("Order:", [2, 3], index=1, key="r_w_order")
                
                w_coeffs = []
                for i in range(order + 1):
                    val = st.number_input(f"a_{i}:", value=1.0 if i == order else 0.0, 
                                         format="%.4f", key=f"r_w_{i}")
                    w_coeffs.append(val)
                st.session_state.w_coeffs = w_coeffs
        
        with col2:
            st.markdown("### 📊 Routh Analysis")
            
            if 'w_coeffs' in st.session_state:
                w_coeffs = st.session_state.w_coeffs
                
                if st.button("🚀 Routh Array", type="primary"):
                    
                    try:
                        routh, stable, first_col = routh_array(w_coeffs)
                        
                        rows_labels = [f"w^{len(w_coeffs)-1-i}" for i in range(routh.shape[0])]
                        routh_df = pd.DataFrame(routh, index=rows_labels)
                        routh_df = routh_df.applymap(lambda x: f"{x:.6f}" if x != 0 else "--")
                        st.dataframe(routh_df, use_container_width=True)
                        
                        st.markdown("#### First Column:")
                        for i, (label, val) in enumerate(zip(rows_labels[:len(first_col)], first_col)):
                            st.write(f"{label}: {val:.6f}")
                        
                        if stable:
                            st.markdown("""
                            <div class="success-box">
                                <h3>✅ STABLE</h3>
                                <p>No sign changes in first column</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="danger-box">
                                <h3>❌ UNSTABLE</h3>
                                <p>Sign changes detected</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")



# ═══════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px;'>
        <p><strong>📚 Digital Control Systems - Weeks 4, 5 & 6</strong></p>
        <p>Dr. Ali Al-Ghanimi | Electrical Engineering | University of Kufa</p>
        <p>📧 ali.alghanimi@uokufa.edu.iq | 🏢 EE Building, Room 301</p>
        <p style='font-size: 0.9rem; color: #666; margin-top: 1rem;'>
            © 2025 University of Kufa | All rights reserved
        </p>
    </div>
""", unsafe_allow_html=True)
