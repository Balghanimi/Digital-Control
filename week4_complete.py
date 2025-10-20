# ═══════════════════════════════════════════════════════════════════
# WEEKS 4-5: INTEGRATED INTERACTIVE DEMONSTRATIONS
# Digital Control Systems | University of Kufa
# Dr. Ali Al-Ghanimi | Electrical Engineering Department
# ═══════════════════════════════════════════════════════════════════

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import control as ct
from datetime import datetime
import json
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Weeks 4-5: Z-Transform & Pulse TF | Dr. Al-Ghanimi",
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
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
    <div class="main-header">
        <h1>🎓 Digital Control Systems</h1>
        <h2>Interactive Demonstrations</h2>
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
         "Week 5: Pulse Transfer Functions"],
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
    else:  # Week 5
        st.markdown("### 🎯 Week 5 Demos")
        demo_selection = st.radio(
            "Select Demo:",
            ["📊 Demo 1: PTF Calculator",
             "🎯 Demo 2: Block Diagram Analyzer", 
             "🔄 Demo 3: Difference Eq ↔ TF",
             "📐 Demo 4: Closed-Loop Systems",
             "🔧 Demo 5: Stability Preview"],
            key="demo_nav_w5"
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
    else:
        with st.expander("Week 5 Objectives"):
            st.markdown("""
            1. ✅ Derive pulse transfer functions
            2. ✅ Analyze block diagrams
            3. ✅ Convert difference equations
            4. ✅ Calculate closed-loop TF
            5. ✅ Preview stability concepts
            """)
    
    st.markdown("---")
    
    st.markdown("### 📖 Resources")
    st.info("""
    **Week 4:** lec 2.pdf, lec 4.pdf, Chakrabortty Ch.2
    
    **Week 5:** DCS.pdf, notes_A2.pdf, Chakrabortty Ch.4
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
        
        with col2:
            st.markdown("### 📈 Visualization")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Time domain
            k = np.arange(0, 20)
            
            if signal_type == "Unit Step":
                x = np.ones_like(k)
                z_transform = "z/(z-1)"
                roc = "|z| > 1"
            elif signal_type == "Unit Ramp":
                x = k
                z_transform = "Tz/(z-1)²"
                roc = "|z| > 1"
            elif "Exponential" in signal_type:
                x = a_val**k
                z_transform = f"z/(z-{a_val:.2f})"
                roc = f"|z| > {abs(a_val):.2f}"
            elif "Sine" in signal_type:
                x = np.sin(freq * k)
                z_transform = "z·sin(ωT)/(z² - 2z·cos(ωT) + 1)"
                roc = "|z| > 1"
            else:  # Cosine
                x = np.cos(freq * k)
                z_transform = "z(z - cos(ωT))/(z² - 2z·cos(ωT) + 1)"
                roc = "|z| > 1"
            
            # Stem plot
            markerline, stemlines, baseline = ax1.stem(k, x, basefmt=' ')
            plt.setp(markerline, color='#667eea', markersize=8)
            plt.setp(stemlines, color='#667eea', linewidth=2)
            ax1.set_xlabel('Sample Index (k)', fontsize=11, fontweight='bold')
            ax1.set_ylabel('x[k]', fontsize=11, fontweight='bold')
            ax1.set_title(f'Time Domain: {signal_type}', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_facecolor('#f8f9fa')
            
            # Z-plane (poles/zeros)
            theta = np.linspace(0, 2*np.pi, 100)
            ax2.plot(np.cos(theta), np.sin(theta), 'b-', linewidth=2, label='Unit Circle')
            
            if "Step" in signal_type or "Ramp" in signal_type:
                ax2.plot([1], [0], 'rx', markersize=15, markeredgewidth=3, label='Pole')
                ax2.plot([0], [0], 'bo', markersize=10, label='Zero')
            elif "Exponential" in signal_type:
                ax2.plot([a_val], [0], 'rx', markersize=15, markeredgewidth=3, label='Pole')
                ax2.plot([0], [0], 'bo', markersize=10, label='Zero')
            
            ax2.axhline(y=0, color='k', linewidth=1)
            ax2.axvline(x=0, color='k', linewidth=1)
            ax2.set_xlabel('Real Part', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Imaginary Part', fontsize=11, fontweight='bold')
            ax2.set_title('Z-Plane', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_aspect('equal')
            ax2.set_xlim(-2, 2)
            ax2.set_ylim(-2, 2)
            ax2.set_facecolor('#f8f9fa')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Results
            st.markdown(f"""
                <div class="success-box">
                    <h4>✅ Z-Transform Result</h4>
                    <p><strong>X(z) = </strong>{z_transform}</p>
                    <p><strong>ROC:</strong> {roc}</p>
                    <p><strong>Signal:</strong> {signal_type}</p>
                </div>
            """, unsafe_allow_html=True)
    
    elif "Demo 2" in demo_selection:
        st.markdown("## 🎯 Demo 2: ROC Visualizer")
        st.info("💡 **Tip:** ROC (Region of Convergence) determines where z-transform is valid!")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Controls")
            signal_type = st.selectbox(
                "Select Signal:",
                ["Causal (right-sided)", "Anti-causal (left-sided)", "Two-sided"]
            )
            pole_real = st.slider("Pole real part:", -1.5, 1.5, 0.8, 0.1)
            pole_imag = st.slider("Pole imaginary part:", -1.5, 1.5, 0.0, 0.1)
        
        with col2:
            st.markdown("### 📈 ROC Visualization")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Z-plane with ROC
            theta = np.linspace(0, 2*np.pi, 100)
            ax1.plot(np.cos(theta), np.sin(theta), 'b-', linewidth=2.5, label='Unit Circle')
            
            pole = pole_real + 1j*pole_imag
            pole_mag = np.abs(pole)
            
            ax1.plot([pole_real], [pole_imag], 'rx', markersize=15, markeredgewidth=3,
                    label=f'Pole: {pole:.2f}')
            
            # Shade ROC
            if 'Causal' in signal_type:
                roc_text = f"|z| > {pole_mag:.2f}"
                for r in np.linspace(max(pole_mag, 0.1), 2.2, 25):
                    ax1.fill(r*np.cos(theta), r*np.sin(theta), 'green', alpha=0.015)
                ax1.text(1.5, 1.5, 'ROC', fontsize=18, fontweight='bold', color='green')
                
            elif 'Anti' in signal_type:
                roc_text = f"|z| < {pole_mag:.2f}"
                for r in np.linspace(0, min(pole_mag, 2.0), 25):
                    ax1.fill(r*np.cos(theta), r*np.sin(theta), 'blue', alpha=0.015)
                ax1.text(0.4, 0.4, 'ROC', fontsize=18, fontweight='bold', color='blue')
                
            else:  # Two-sided
                roc_text = f"{pole_mag*0.5:.2f} < |z| < {pole_mag*1.5:.2f}"
                for r in np.linspace(max(pole_mag*0.5, 0.1), pole_mag*1.5, 25):
                    ax1.fill(r*np.cos(theta), r*np.sin(theta), 'orange', alpha=0.015)
                ax1.text(1.0, 1.0, 'ROC', fontsize=18, fontweight='bold', color='orange')
            
            ax1.set_xlabel('Real Part', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Imaginary Part', fontsize=11, fontweight='bold')
            ax1.set_title('Z-Plane: ROC', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='k', linewidth=1)
            ax1.axvline(x=0, color='k', linewidth=1)
            ax1.set_aspect('equal')
            ax1.set_xlim(-2.3, 2.3)
            ax1.set_ylim(-2.3, 2.3)
            ax1.legend()
            ax1.set_facecolor('#f8f9fa')
            
            # Time domain
            n = np.arange(-10, 30)
            if 'Causal' in signal_type:
                x = np.where(n >= 0, pole**n, 0)
            elif 'Anti' in signal_type:
                x = np.where(n < 0, pole**(-n), 0)
            else:
                x = 0.6**(np.abs(n))
            
            x = np.real(x)
            
            markerline, stemlines, baseline = ax2.stem(n, x, basefmt=' ')
            plt.setp(markerline, color='#667eea', markersize=7)
            plt.setp(stemlines, color='#667eea', linewidth=2)
            ax2.set_xlabel('Sample Index (n)', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Amplitude x[n]', fontsize=11, fontweight='bold')
            ax2.set_title('Time-Domain Signal', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='k', linewidth=1)
            ax2.axvline(x=0, color='red', linewidth=1.5, linestyle='--', alpha=0.6)
            ax2.set_facecolor('#f8f9fa')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Analysis
            stable = pole_mag < 1 and 'Causal' in signal_type
            
            if stable:
                st.markdown(f"""
                    <div class="success-box">
                        <h4>✅ ROC Analysis</h4>
                        <p><strong>Pole:</strong> z = {pole:.3f} (|z| = {pole_mag:.3f})</p>
                        <p><strong>ROC:</strong> {roc_text}</p>
                        <p><strong>Signal:</strong> {signal_type}</p>
                        <p><strong>Stability:</strong> ✅ STABLE</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="warning-box">
                        <h4>⚠️ ROC Analysis</h4>
                        <p><strong>Pole:</strong> z = {pole:.3f} (|z| = {pole_mag:.3f})</p>
                        <p><strong>ROC:</strong> {roc_text}</p>
                        <p><strong>Signal:</strong> {signal_type}</p>
                        <p><strong>Stability:</strong> ⚠️ Check ROC</p>
                    </div>
                """, unsafe_allow_html=True)
    
    # Add simplified Demo 3-5 for Week 4
    else:
        st.markdown(f"## {demo_selection}")
        st.info("🚧 Demo in progress. Full implementation similar to Demos 1-2.")

# ═══════════════════════════════════════════════════════════════════
# WEEK 5 DEMOS
# ═══════════════════════════════════════════════════════════════════

else:  # Week 5
    
    if "Demo 1" in demo_selection:
        st.markdown("## 📊 Demo 1: Pulse Transfer Function Calculator")
        st.markdown("Convert continuous G(s) with ZOH to discrete G(z)")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Controls")
            
            system_type = st.selectbox(
                "Select System:",
                ["First-Order: a/(s+a)",
                 "Second-Order: ω²/(s²+2ζωs+ω²)",
                 "Integrator: 1/s",
                 "Double Integrator: 1/s²",
                 "Custom"]
            )
            
            if "First-Order" in system_type:
                a = st.slider("Pole location (a):", 0.5, 10.0, 2.0, 0.5)
                T = st.slider("Sampling period (T):", 0.01, 1.0, 0.1, 0.01)
                
                # Calculate G(z)
                pole_z = np.exp(-a*T)
                gain_z = (1 - pole_z)
                
                st.markdown(f"""
                **Continuous:**  
                $$G_p(s) = \\frac{{{a}}}{s+{a}}$$
                
                **Discrete (with ZOH):**  
                $$G(z) = \\frac{{{gain_z:.4f}}}{{z-{pole_z:.4f}}}$$
                """)
                
            elif "Integrator" in system_type:
                T = st.slider("Sampling period (T):", 0.01, 1.0, 0.1, 0.01)
                
                pole_z = 1.0
                gain_z = T
                
                st.markdown(f"""
                **Continuous:**  
                $$G_p(s) = \\frac{{1}}{{s}}$$
                
                **Discrete (with ZOH):**  
                $$G(z) = \\frac{{{gain_z:.4f}z}}{{(z-1)^2}}$$
                """)
        
        with col2:
            st.markdown("### 📈 Pole-Zero Map")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # S-plane
            if "First-Order" in system_type:
                ax1.plot([-a], [0], 'rx', markersize=15, markeredgewidth=3, label=f'Pole: s=-{a}')
            else:
                ax1.plot([0], [0], 'rx', markersize=15, markeredgewidth=3, label='Pole: s=0')
            
            ax1.axhline(y=0, color='k', linewidth=1)
            ax1.axvline(x=0, color='k', linewidth=1.5, linestyle='--', color='red')
            ax1.fill_between([-15, 0], [-10, -10], [10, 10], alpha=0.2, color='green', label='Stable Region')
            ax1.set_xlabel('Real (σ)', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Imaginary (jω)', fontsize=11, fontweight='bold')
            ax1.set_title('S-Plane (Continuous)', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_xlim(-15, 5)
            ax1.set_ylim(-10, 10)
            ax1.set_facecolor('#f8f9fa')
            
            # Z-plane
            theta = np.linspace(0, 2*np.pi, 100)
            ax2.plot(np.cos(theta), np.sin(theta), 'b-', linewidth=2.5, label='Unit Circle')
            
            if "First-Order" in system_type:
                ax2.plot([pole_z], [0], 'rx', markersize=15, markeredgewidth=3, label=f'Pole: z={pole_z:.3f}')
                ax2.plot([0], [0], 'bo', markersize=10, label='Zero: z=0')
            else:
                ax2.plot([1], [0], 'rx', markersize=15, markeredgewidth=3, label='Pole: z=1')
            
            # Shade stable region
            for r in np.linspace(0, 1, 25):
                ax2.fill(r*np.cos(theta), r*np.sin(theta), 'green', alpha=0.01)
            ax2.text(0.5, 0.5, 'STABLE\n|z|<1', fontsize=12, fontweight='bold', 
                    color='green', ha='center')
            
            ax2.axhline(y=0, color='k', linewidth=1)
            ax2.axvline(x=0, color='k', linewidth=1)
            ax2.set_xlabel('Real Part', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Imaginary Part', fontsize=11, fontweight='bold')
            ax2.set_title('Z-Plane (Discrete)', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_aspect('equal')
            ax2.set_xlim(-1.5, 1.5)
            ax2.set_ylim(-1.5, 1.5)
            ax2.set_facecolor('#f8f9fa')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Stability check
            if "First-Order" in system_type:
                is_stable = pole_z < 1
                st.markdown(f"""
                    <div class="{'success' if is_stable else 'warning'}-box">
                        <h4>{'✅' if is_stable else '⚠️'} Stability Analysis</h4>
                        <p><strong>Discrete Pole:</strong> z = {pole_z:.4f}</p>
                        <p><strong>|z| = </strong>{abs(pole_z):.4f}</p>
                        <p><strong>Stability:</strong> {'STABLE' if is_stable else 'Check conditions'} 
                        (pole {'inside' if is_stable else 'on/outside'} unit circle)</p>
                    </div>
                """, unsafe_allow_html=True)
    
    elif "Demo 2" in demo_selection:
        st.markdown("## 🎯 Demo 2: Block Diagram Analyzer")
        st.markdown("Analyze cascaded systems with/without samplers")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Configuration")
            
            config = st.radio(
                "Cascaded elements:",
                ["Separated by sampler", "NOT separated"]
            )
            
            a1 = st.slider("G₁: pole location (a₁):", 0.5, 5.0, 1.0, 0.5)
            a2 = st.slider("G₂: pole location (a₂):", 0.5, 5.0, 3.0, 0.5)
            T = st.slider("Sampling period (T):", 0.05, 0.5, 0.1, 0.05)
            
            st.markdown(f"""
            **System:**  
            $$G_1(s) = \\frac{{1}}{{s+{a1}}}$$  
            $$G_2(s) = \\frac{{1}}{{s+{a2}}}$$
            """)
        
        with col2:
            st.markdown("### 📊 Block Diagram & Analysis")
            
            # Visual representation
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.axis('off')
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 2)
            
            # Draw blocks
            if "Separated" in config:
                ax.add_patch(plt.Rectangle((1, 0.7), 1.5, 0.6, fc='lightblue', ec='black', lw=2))
                ax.text(1.75, 1.0, '$G_1(s)$', ha='center', va='center', fontsize=12, fontweight='bold')
                
                ax.add_patch(plt.Circle((3.5, 1.0), 0.25, fc='white', ec='blue', lw=2))
                ax.text(3.5, 1.0, 'T', ha='center', va='center', fontsize=10, fontweight='bold', color='blue')
                
                ax.add_patch(plt.Rectangle((5, 0.7), 1.5, 0.6, fc='lightgreen', ec='black', lw=2))
                ax.text(5.75, 1.0, '$G_2(s)$', ha='center', va='center', fontsize=12, fontweight='bold')
                
                ax.arrow(0.5, 1.0, 0.4, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')
                ax.arrow(2.6, 1.0, 0.6, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')
                ax.arrow(3.8, 1.0, 1.0, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')
                ax.arrow(6.6, 1.0, 0.4, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')
                
                result_text = "G(z) = G₁(z) · G₂(z)"
                
            else:
                ax.add_patch(plt.Rectangle((2, 0.7), 1.5, 0.6, fc='lightblue', ec='black', lw=2))
                ax.text(2.75, 1.0, '$G_1(s)$', ha='center', va='center', fontsize=12, fontweight='bold')
                
                ax.add_patch(plt.Rectangle((4, 0.7), 1.5, 0.6, fc='lightgreen', ec='black', lw=2))
                ax.text(4.75, 1.0, '$G_2(s)$', ha='center', va='center', fontsize=12, fontweight='bold')
                
                ax.add_patch(plt.Circle((6.5, 1.0), 0.25, fc='white', ec='blue', lw=2))
                ax.text(6.5, 1.0, 'T', ha='center', va='center', fontsize=10, fontweight='bold', color='blue')
                
                ax.arrow(1.0, 1.0, 0.9, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')
                ax.arrow(3.6, 1.0, 0.3, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')
                ax.arrow(5.6, 1.0, 0.7, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')
                ax.arrow(6.8, 1.0, 0.4, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')
                
                result_text = "G(z) = Z{G₁(s)·G₂(s)}"
            
            st.pyplot(fig)
            
            # Calculate transfer functions
            pole1_z = np.exp(-a1*T)
            pole2_z = np.exp(-a2*T)
            
            if "Separated" in config:
                # Product of individual TFs
                G1_z = f"(1-e^(-{a1}T))/(z-e^(-{a1}T))"
                G2_z = f"(1-e^(-{a2}T))/(z-e^(-{a2}T))"
                
                st.markdown(f"""
                <div class="success-box">
                    <h4>✅ Result: Separated by Sampler</h4>
                    <p><strong>Rule:</strong> {result_text}</p>
                    <p><strong>G₁(z) =</strong> {(1-pole1_z):.4f}/(z-{pole1_z:.4f})</p>
                    <p><strong>G₂(z) =</strong> {(1-pole2_z):.4f}/(z-{pole2_z:.4f})</p>
                    <p><strong>Poles:</strong> z = {pole1_z:.4f}, {pole2_z:.4f}</p>
                    <p>Both inside unit circle → <strong>STABLE</strong> ✓</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-box">
                    <h4>⚠️ Result: NOT Separated</h4>
                    <p><strong>Rule:</strong> {result_text}</p>
                    <p>Must first compute: G₁(s)·G₂(s) = 1/[(s+{a1})(s+{a2})]</p>
                    <p>Then apply Z-transform with ZOH</p>
                    <p><strong>Result will be DIFFERENT from separated case!</strong></p>
                </div>
                """, unsafe_allow_html=True)
    
    elif "Demo 3" in demo_selection:
        st.markdown("## 🔄 Demo 3: Difference Equation ↔ Transfer Function")
        st.markdown("Convert between time-domain and z-domain representations")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Input")
            
            conversion_dir = st.radio("Direction:", 
                ["Diff Eq → TF", "TF → Diff Eq"])
            
            if "Diff Eq" in conversion_dir:
                st.markdown("**Enter coefficients:**")
                a1 = st.number_input("a₁ (coef of y(k-1)):", -2.0, 2.0, 0.8, 0.1)
                a2 = st.number_input("a₂ (coef of y(k-2)):", -2.0, 2.0, -0.2, 0.1)
                b0 = st.number_input("b₀ (coef of x(k)):", 0.0, 5.0, 1.0, 0.1)
                b1 = st.number_input("b₁ (coef of x(k-1)):", -2.0, 2.0, 0.5, 0.1)
                
                st.markdown(f"""
                **Difference Equation:**  
                $$y(k) - {a1}y(k-1) - {a2}y(k-2) = {b0}x(k) + {b1}x(k-1)$$
                """)
        
        with col2:
            st.markdown("### 📊 Conversion Result")
            
            if "Diff Eq" in conversion_dir:
                # Calculate transfer function
                num = [b0, b1]
                den = [1, -a1, -a2]
                
                # Find poles
                poles = np.roots(den)
                
                st.markdown(f"""
                **Transfer Function:**  
                $$G(z) = \\frac{{{b0}z + {b1}}}{{z^2 - {a1}z - {a2}}}$$
                """)
                
                # Pole-zero plot
                fig, ax = plt.subplots(figsize=(8, 8))
                
                theta = np.linspace(0, 2*np.pi, 100)
                ax.plot(np.cos(theta), np.sin(theta), 'b-', linewidth=2.5, label='Unit Circle')
                
                # Plot poles
                ax.plot(poles.real, poles.imag, 'rx', markersize=15, markeredgewidth=3, label='Poles')
                
                # Shade stable region
                for r in np.linspace(0, 1, 25):
                    ax.fill(r*np.cos(theta), r*np.sin(theta), 'green', alpha=0.01)
                
                ax.axhline(y=0, color='k', linewidth=1)
                ax.axvline(x=0, color='k', linewidth=1)
                ax.set_xlabel('Real Part', fontsize=12, fontweight='bold')
                ax.set_ylabel('Imaginary Part', fontsize=12, fontweight='bold')
                ax.set_title('Pole-Zero Map', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=11)
                ax.set_aspect('equal')
                ax.set_xlim(-1.5, 1.5)
                ax.set_ylim(-1.5, 1.5)
                ax.set_facecolor('#f8f9fa')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Stability analysis
                all_stable = all(np.abs(poles) < 1)
                
                st.markdown(f"""
                <div class="{'success' if all_stable else 'warning'}-box">
                    <h4>{'✅' if all_stable else '⚠️'} Stability Analysis</h4>
                    <p><strong>Poles:</strong></p>
                    <ul>
                        <li>z₁ = {poles[0]:.4f} (|z₁| = {abs(poles[0]):.4f})</li>
                        <li>z₂ = {poles[1]:.4f} (|z₂| = {abs(poles[1]):.4f})</li>
                    </ul>
                    <p><strong>Stability:</strong> {'ALL poles inside unit circle → STABLE ✓' if all_stable else 'Some poles outside → UNSTABLE ✗'}</p>
                </div>
                """, unsafe_allow_html=True)
    
    elif "Demo 4" in demo_selection:
        st.markdown("## 📐 Demo 4: Closed-Loop System Analysis")
        st.markdown("Analyze feedback systems and characteristic equations")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ System Parameters")
            
            K = st.slider("Gain (K):", 0.1, 5.0, 1.0, 0.1)
            pole1 = st.slider("Open-loop pole 1:", 0.1, 0.95, 0.5, 0.05)
            pole2 = st.slider("Open-loop pole 2:", 0.1, 0.95, 0.7, 0.05)
            
            st.markdown(f"""
            **Open-Loop Transfer Function:**  
            $$G(z) = \\frac{{K}}{{(z-{pole1})(z-{pole2})}}$$
            
            **Feedback:** Unity (H(z) = 1)
            """)
        
        with col2:
            st.markdown("### 📊 Root Locus & Analysis")
            
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
            ax.set_title(f'Pole Movement (K={K})', fontsize=14, fontweight='bold')
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
            $$1 + G(z) = 0$$  
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
    
    elif "Demo 5" in demo_selection:
        st.markdown("## 🔧 Demo 5: Stability Preview - Unit Circle Criterion")
        st.markdown("Preview of Week 6: Understanding stability in the z-plane")
        
        st.info("""
        💡 **Preview for Week 6:** 
        
        For discrete-time systems, stability is determined by pole locations relative to the **unit circle**:
        - **Stable:** ALL poles inside unit circle (|z| < 1)
        - **Unstable:** Any pole outside unit circle (|z| > 1)  
        - **Marginally Stable:** Poles on unit circle (|z| = 1)
        
        Next week we'll learn **Jury Test** and **Routh Criterion** to check this without solving for poles!
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Test Different Systems")
            
            num_poles = st.slider("Number of poles:", 1, 4, 2, 1)
            
            poles = []
            for i in range(num_poles):
                st.markdown(f"**Pole {i+1}:**")
                real = st.slider(f"Real part {i+1}:", -1.5, 1.5, 0.5, 0.1, key=f"real{i}")
                imag = st.slider(f"Imag part {i+1}:", -1.5, 1.5, 0.0, 0.1, key=f"imag{i}")
                poles.append(real + 1j*imag)
        
        with col2:
            st.markdown("### 📊 Stability Visualization")
            
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Unit circle
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta), 'b-', linewidth=3, label='Unit Circle (|z|=1)')
            
            # Shade regions
            for r in np.linspace(0, 1, 30):
                ax.fill(r*np.cos(theta), r*np.sin(theta), 'green', alpha=0.008)
            ax.text(0, 0.6, 'STABLE\nREGION\n|z| < 1', fontsize=14, fontweight='bold',
                    color='darkgreen', ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            ax.text(1.8, 1.5, 'UNSTABLE\nREGION\n|z| > 1', fontsize=14, fontweight='bold',
                    color='red', ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Plot poles
            for i, pole in enumerate(poles):
                mag = np.abs(pole)
                color = 'green' if mag < 1 else ('orange' if np.isclose(mag, 1) else 'red')
                ax.plot([pole.real], [pole.imag], 'x', color=color, markersize=20, markeredgewidth=4,
                        label=f'Pole {i+1}: {pole:.3f} (|z|={mag:.3f})')
            
            ax.axhline(y=0, color='k', linewidth=1)
            ax.axvline(x=0, color='k', linewidth=1)
            ax.set_xlabel('Real Part', fontsize=13, fontweight='bold')
            ax.set_ylabel('Imaginary Part', fontsize=13, fontweight='bold')
            ax.set_title('Z-Plane Stability Map', fontsize=15, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc='upper left')
            ax.set_aspect('equal')
            ax.set_xlim(-2.2, 2.2)
            ax.set_ylim(-2.2, 2.2)
            ax.set_facecolor('#f8f9fa')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Detailed analysis
            all_stable = all(np.abs(p) < 1 for p in poles)
            marginal = any(np.isclose(np.abs(p), 1) for p in poles)
            
            if all_stable and not marginal:
                st.markdown(f"""
                <div class="success-box">
                    <h4>✅ System is STABLE</h4>
                    <p><strong>All {num_poles} poles are inside the unit circle!</strong></p>
                    <p>The system output will decay to zero (or converge to steady-state).</p>
                    <p><strong>Next week:</strong> Learn Jury Test to verify this algebraically!</p>
                </div>
                """, unsafe_allow_html=True)
            elif marginal:
                st.markdown(f"""
                <div class="warning-box">
                    <h4>⚠️ System is MARGINALLY STABLE</h4>
                    <p>One or more poles are <strong>on</strong> the unit circle (|z| = 1)</p>
                    <p>The system will have sustained oscillations (no decay, no growth).</p>
                    <p><strong>Next week:</strong> Learn to find marginal stability conditions!</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-box">
                    <h4>❌ System is UNSTABLE</h4>
                    <p>One or more poles are <strong>outside</strong> the unit circle (|z| > 1)</p>
                    <p>The system output will grow without bound!</p>
                    <p><strong>Solution:</strong> Adjust controller gain or redesign controller.</p>
                    <p><strong>Next week:</strong> Find stable gain range using stability tests!</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Comparison table
            st.markdown("### 📋 Continuous vs. Discrete Stability")
            
            comparison_data = {
                "": ["Stable Region", "Boundary", "Unstable Region"],
                "S-Plane (Continuous)": ["Re(s) < 0 (LHP)", "Re(s) = 0 (jω axis)", "Re(s) > 0 (RHP)"],
                "Z-Plane (Discrete)": ["|z| < 1 (Inside circle)", "|z| = 1 (On circle)", "|z| > 1 (Outside circle)"]
            }
            
            import pandas as pd
            df = pd.DataFrame(comparison_data)
            st.table(df)
            
            st.markdown("""
            **Mapping:** $z = e^{sT}$  
            - LHP in s-plane → Inside unit circle in z-plane  
            - jω axis in s-plane → Unit circle in z-plane  
            - RHP in s-plane → Outside unit circle in z-plane
            """)

# ═══════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px;'>
        <p><strong>📚 Digital Control Systems - Weeks 4 & 5</strong></p>
        <p>Dr. Ali Al-Ghanimi | Electrical Engineering | University of Kufa</p>
        <p>📧 ali.alghanimi@uokufa.edu.iq | 🏢 EE Building, Room 301</p>
        <p style='font-size: 0.9rem; color: #666; margin-top: 1rem;'>
            © 2025 University of Kufa | All rights reserved
        </p>
    </div>
""", unsafe_allow_html=True)
