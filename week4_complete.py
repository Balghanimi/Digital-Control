"""
═══════════════════════════════════════════════════════════════════
DIGITAL CONTROL SYSTEMS - INTEGRATED INTERACTIVE DEMONSTRATIONS
University of Kufa | Electrical Engineering Department
Dr. Ali Al-Ghanimi | Academic Year 2025

Weeks 4-6 Complete Coverage:
- Week 4: Z-Transform & Inverse Z-Transform
- Week 5: Pulse Transfer Functions  
- Week 6: Stability Analysis I (Jury Test & Routh Criterion)

Source References:
- Chakrabortty et al., "Digital Control System Analysis & Design"
- DigitalControlTextBook.pdf (Ch. 3-4)
- DCS.pdf, lec_2.pdf, lec_4.pdf
- notes_A2_DiscreteSystems.pdf
═══════════════════════════════════════════════════════════════════
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from scipy import signal
import control as ct
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Digital Control Systems | Weeks 4-6 | Dr. Al-Ghanimi",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:alih.alghanimi@uokufa.edu.iq',
        'About': 'Digital Control Systems - University of Kufa'
    }
)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
if 'view_count' not in st.session_state:
    st.session_state.view_count = 0

# Custom CSS for enhanced UI
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .demo-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
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
    .formula-box {
        background: #e8e9ea;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #6c757d;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
    }
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS - WEEK 4
# ═══════════════════════════════════════════════════════════════════

def z_transform(signal_type="step", a=1.0):
    """Calculate Z-transform for common signals"""
    if signal_type == "step":
        return f"Z[u[n]] = z/(z-1)", np.array([1, -1])
    elif signal_type == "exponential":
        return f"Z[a^n u[n]] = z/(z-{a:.2f})", np.array([1, -a])
    elif signal_type == "ramp":
        return f"Z[n u[n]] = z/(z-1)^2", np.array([1, -2, 1])

def draw_roc(ax, poles, roc_type="exterior"):
    """Draw Region of Convergence on complex plane"""
    circle = plt.Circle((0, 0), 1, color='blue', fill=False, linewidth=2, linestyle='--')
    ax.add_patch(circle)
    
    if roc_type == "exterior":
        max_pole = max(abs(p) for p in poles)
        roc_circle = plt.Circle((0, 0), max_pole, color='green', fill=True, alpha=0.2)
        ax.add_patch(roc_circle)
    
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('Real', fontweight='bold')
    ax.set_ylabel('Imaginary', fontweight='bold')

def partial_fractions(num, den):
    """Decompose transfer function into partial fractions"""
    residues, poles, k = signal.residue(num, den)
    return residues, poles, k

def inverse_z_transform_methods(num, den, n_samples=20):
    """Compare different inverse Z-transform methods"""
    # Method 1: Partial Fractions
    residues, poles, _ = partial_fractions(num, den)
    
    # Method 2: Long Division
    sys = signal.TransferFunction(num, den, dt=True)
    t, y_impulse = signal.dimpulse(sys, n=n_samples)
    
    # Method 3: Direct calculation
    n = np.arange(n_samples)
    y_direct = np.zeros(n_samples)
    for i, (r, p) in enumerate(zip(residues, poles)):
        y_direct += r * (p ** n)
    
    return n, y_impulse[0].flatten(), y_direct

# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS - WEEK 5
# ═══════════════════════════════════════════════════════════════════

def diff_eq_to_tf(a_coeffs, b_coeffs):
    """Convert difference equation to transfer function"""
    H_z = signal.TransferFunction(b_coeffs, a_coeffs, dt=True)
    return H_z

def block_diagram_reduction(G1, G2, connection="series"):
    """Reduce block diagrams"""
    if connection == "series":
        return signal.TransferFunction(
            np.polymul(G1.num, G2.num),
            np.polymul(G1.den, G2.den),
            dt=True
        )
    elif connection == "parallel":
        num = np.polymul(G1.num, G2.den) + np.polymul(G2.num, G1.den)
        den = np.polymul(G1.den, G2.den)
        return signal.TransferFunction(num, den, dt=True)
    elif connection == "feedback":
        num = np.polymul(G1.num, G2.den)
        den = np.polymul(G1.den, G2.den) + np.polymul(G1.num, G2.num)
        return signal.TransferFunction(num, den, dt=True)

# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS - WEEK 6 (ENHANCED)
# ═══════════════════════════════════════════════════════════════════

def evaluate_polynomial(coeffs, z_value):
    """Evaluate polynomial Q(z) at given z value"""
    return np.sum(coeffs * (z_value ** np.arange(len(coeffs))))

def jury_test(coeffs):
    """
    Comprehensive Jury stability test
    Source: DigitalControlTextBook.pdf, Section 4.5, pp. 104-109
    """
    n = len(coeffs) - 1
    a = np.array(coeffs, dtype=float)
    
    details = {
        'conditions': [],
        'jury_table': [],
        'failed_at': None
    }
    
    # Condition 1: Q(1) > 0
    Q1 = np.sum(a)
    cond1_pass = Q1 > 0
    details['conditions'].append({
        'number': 1,
        'description': 'Q(1) > 0',
        'value': Q1,
        'pass': cond1_pass
    })
    
    if not cond1_pass:
        details['failed_at'] = 1
        return False, details
    
    # Condition 2: (-1)^n * Q(-1) > 0
    Q_minus1 = evaluate_polynomial(a, -1)
    cond2_value = ((-1)**n) * Q_minus1
    cond2_pass = cond2_value > 0
    details['conditions'].append({
        'number': 2,
        'description': f'(-1)^{n} Q(-1) > 0',
        'value': cond2_value,
        'pass': cond2_pass
    })
    
    if not cond2_pass:
        details['failed_at'] = 2
        return False, details
    
    # Condition 3: |a0| < an
    cond3_pass = abs(a[0]) < abs(a[n])
    details['conditions'].append({
        'number': 3,
        'description': '|a0| < |an|',
        'value': f'|{a[0]:.4f}| < |{a[n]:.4f}|',
        'pass': cond3_pass
    })
    
    if not cond3_pass:
        details['failed_at'] = 3
        return False, details
    
    # Build Jury table for n >= 2
    if n >= 2:
        # Initialize table rows
        table_rows = []
        table_rows.append(a.copy())  # Row 1: coefficients
        table_rows.append(a[::-1].copy())  # Row 2: reversed coefficients
        
        current_coeffs = a.copy()
        condition_number = 4
        
        for row_idx in range(n - 1):
            # Create new row using determinants
            new_row_len = len(current_coeffs) - 1
            new_row = np.zeros(new_row_len)
            
            for j in range(new_row_len):
                # Calculate determinant: |a0 a_{n-j}|
                #                        |an a_j    |
                det = current_coeffs[0] * current_coeffs[-(j+2)] - current_coeffs[-1] * current_coeffs[j]
                new_row[j] = det
            
            if new_row_len > 0:
                # Check condition |b0| > |b_{n-1}|
                cond_pass = abs(new_row[0]) > abs(new_row[-1])
                details['conditions'].append({
                    'number': condition_number,
                    'description': f'Row {row_idx + 3}: |first| > |last|',
                    'value': f'|{new_row[0]:.4f}| > |{new_row[-1]:.4f}|',
                    'pass': cond_pass
                })
                
                if not cond_pass:
                    details['failed_at'] = condition_number
                    return False, details
                
                condition_number += 1
            
            table_rows.append(new_row.copy())
            if new_row_len > 1:
                table_rows.append(new_row[::-1].copy())
            
            current_coeffs = new_row.copy()
            
            if len(current_coeffs) <= 1:
                break
        
        details['jury_table'] = table_rows
    
    return True, details

def bilinear_transform(coeffs_z):
    """
    Apply bilinear transformation w = (z-1)/(z+1) or z = (1+w)/(1-w)
    Source: Chakrabortty Ch.7, pp.234-236
    
    For a polynomial Q(z) = sum(a_i * z^i), substitute z = (1+w)/(1-w)
    and expand to get Q(w).
    """
    # Ensure all coefficients are float type
    coeffs_z = [float(c) for c in coeffs_z]
    n = len(coeffs_z) - 1
    
    # For simple 2nd order case (most common)
    if n == 2:
        a0, a1, a2 = coeffs_z
        # Q(z) = a2*z^2 + a1*z + a0
        # Substitute z = (1+w)/(1-w) and expand
        # After algebra, Q(w) = b2*w^2 + b1*w + b0
        b0 = float(a2 + a1 + a0)
        b1 = float(2*(a2 - a0))
        b2 = float(a2 - a1 + a0)
        
        # Normalize
        if abs(b2) > 1e-10:
            return np.array([b0/b2, b1/b2, 1.0])
        else:
            return np.array([b0, b1, b2])
    
    # For 3rd order
    elif n == 3:
        a0, a1, a2, a3 = coeffs_z
        # After substitution and expansion
        b0 = float(a3 + a2 + a1 + a0)
        b1 = float(3*a3 + a2 - a1 - 3*a0)
        b2 = float(3*a3 - a2 - a1 + 3*a0)
        b3 = float(a3 - a2 + a1 - a0)
        
        # Normalize
        if abs(b3) > 1e-10:
            return np.array([b0/b3, b1/b3, b2/b3, 1.0])
        else:
            return np.array([b0, b1, b2, b3])
    
    # For higher orders, use general formula
    else:
        from scipy.special import comb
        coeffs_w = np.zeros(n + 1)
        
        for i in range(n + 1):
            for k in range(n + 1):
                # Coefficient contribution from binomial expansion
                for j in range(k + 1):
                    if i == n - k + 2*j:
                        coeffs_w[i] += coeffs_z[k] * comb(k, j, exact=True) * ((-1)**(k-j))
        
        # Normalize
        if coeffs_w[-1] != 0:
            coeffs_w = coeffs_w / coeffs_w[-1]
        
        return coeffs_w

def routh_array(coeffs):
    """
    Construct Routh array for stability analysis
    Source: Chakrabortty Ch.7, pp.236-239
    """
    n = len(coeffs) - 1
    
    # Initialize array with proper size
    cols = (n + 2) // 2 + 1
    routh = np.zeros((n + 1, cols))
    
    # First row: coefficients with even indices (in reverse order)
    for i in range(0, len(coeffs), 2):
        if i // 2 < cols:
            routh[0, i // 2] = coeffs[n - i]
    
    # Second row: coefficients with odd indices (in reverse order)
    if n > 0:
        for i in range(1, len(coeffs), 2):
            if i // 2 < cols:
                routh[1, i // 2] = coeffs[n - i]
    
    # Calculate remaining rows
    for i in range(2, n + 1):
        for j in range(cols - 1):
            if routh[i-1, 0] != 0:
                if j + 1 < cols:
                    numerator = routh[i-2, 0] * routh[i-1, j+1] - routh[i-2, j+1] * routh[i-1, 0]
                    routh[i, j] = numerator / routh[i-1, 0]
            else:
                # Handle zero in first column by using small epsilon
                eps = 1e-10
                if j + 1 < cols:
                    numerator = routh[i-2, 0] * routh[i-1, j+1] - routh[i-2, j+1] * eps
                    routh[i, j] = numerator / eps
    
    return routh

def check_routh_stability(routh):
    """Check stability from Routh array"""
    first_col = routh[:, 0]
    
    # Count sign changes in first column
    sign_changes = 0
    prev_sign = np.sign(first_col[0]) if first_col[0] != 0 else 1
    
    for val in first_col[1:]:
        if val != 0:
            curr_sign = np.sign(val)
            if prev_sign * curr_sign < 0:
                sign_changes += 1
            prev_sign = curr_sign
    
    return sign_changes == 0, sign_changes

def find_stability_range(char_poly_coeffs, K_var_index, K_range):
    """Find stable range of parameter K"""
    stable_range = []
    
    for K in K_range:
        test_coeffs = char_poly_coeffs.copy()
        test_coeffs[K_var_index] = K
        
        is_stable, _ = jury_test(test_coeffs)
        if is_stable:
            stable_range.append(K)
    
    if stable_range:
        return min(stable_range), max(stable_range)
    return None, None

def draw_unit_circle(ax, title="Unit Circle"):
    """Draw unit circle for stability visualization"""
    circle = plt.Circle((0, 0), 1, color='blue', fill=False, linewidth=2)
    ax.add_patch(circle)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Real', fontsize=12, fontweight='bold')
    ax.set_ylabel('Imaginary', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')

# ═══════════════════════════════════════════════════════════════════
# MAIN HEADER
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
# SIDEBAR NAVIGATION
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
    elif "Week 5" in week_selection:
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
            
            **References:** lec_2.pdf, lec_4.pdf, Chakrabortty Ch.2
            """)
    elif "Week 5" in week_selection:
        with st.expander("Week 5 Objectives"):
            st.markdown("""
            1. ✅ Derive pulse transfer functions
            2. ✅ Analyze block diagrams
            3. ✅ Convert difference equations
            4. ✅ Calculate closed-loop TF
            5. ✅ Compare open vs closed loop
            
            **References:** DCS.pdf, notes_A2.pdf, Chakrabortty Ch.4
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
            
            **References:** Chakrabortty Ch.7, DigitalControlTextBook Ch.4
            """)
    
    st.markdown("---")
    
    st.markdown("### 💬 Contact")
    st.markdown("""
    📧 **Email:** ali.alghanimi@uokufa.edu.iq  
    🏢 **Office:** EE Building, Room 115  
    ⏰ **Office Hours:** Sun-Thu 10:00-12:00
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 0.5rem;'>
        <small>© 2025 University of Kufa</small>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# MAIN CONTENT AREA
# ═══════════════════════════════════════════════════════════════════

# Week 4 Demos
if "Week 4" in week_selection:
    
    if "Demo 1" in demo_selection:
        st.markdown("## 📊 Demo 1: Z-Transform Calculator")
        st.markdown("Calculate Z-transforms of common discrete-time signals")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Signal Selection")
            signal_type = st.selectbox(
                "Select signal type:",
                ["Unit Step", "Exponential", "Ramp", "Impulse"],
                key="z_sig_type"
            )
            
            if signal_type == "Exponential":
                a = st.slider("Parameter a:", 0.1, 2.0, 0.5, 0.1, key="z_exp_param")
            else:
                a = 1.0
            
            n_samples = st.slider("Samples to plot:", 10, 50, 20, key="z_samples")
        
        with col2:
            st.markdown("### 📈 Results")
            
            if signal_type == "Unit Step":
                z_expr = "z/(z-1)"
                poles = [1.0]
                signal_vals = np.ones(n_samples)
            elif signal_type == "Exponential":
                z_expr = f"z/(z-{a:.2f})"
                poles = [a]
                signal_vals = a ** np.arange(n_samples)
            elif signal_type == "Ramp":
                z_expr = "z/(z-1)²"
                poles = [1.0, 1.0]
                signal_vals = np.arange(n_samples)
            else:  # Impulse
                z_expr = "1"
                poles = []
                signal_vals = np.zeros(n_samples)
                signal_vals[0] = 1
            
            st.latex(f"Z\\{{x[n]\\}} = {z_expr}")
            
            # Plot signal and pole-zero
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Time domain signal
            ax1.stem(range(n_samples), signal_vals, basefmt=' ')
            ax1.set_xlabel('n', fontweight='bold')
            ax1.set_ylabel('x[n]', fontweight='bold')
            ax1.set_title('Time Domain Signal', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Pole-zero plot
            draw_unit_circle(ax2, "Pole-Zero Plot")
            for p in poles:
                ax2.plot(p, 0, 'x', markersize=12, color='red', markeredgewidth=2)
            ax2.plot(0, 0, 'o', markersize=8, color='blue')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.success(f"✅ Z-transform calculated for {signal_type.lower()} signal")
    
    elif "Demo 2" in demo_selection:
        st.markdown("## 🎯 Demo 2: ROC Visualizer")
        st.markdown("Visualize Region of Convergence for different pole configurations")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Pole Configuration")
            
            pole_config = st.selectbox(
                "Select configuration:",
                ["Single Real Pole", "Complex Conjugate Pair", "Multiple Poles"],
                key="roc_config"
            )
            
            if pole_config == "Single Real Pole":
                pole_val = st.slider("Pole location:", -2.0, 2.0, 0.8, 0.1, key="roc_single")
                poles = [pole_val]
            elif pole_config == "Complex Conjugate Pair":
                mag = st.slider("Magnitude:", 0.1, 1.5, 0.9, 0.1, key="roc_mag")
                angle = st.slider("Angle (degrees):", 0, 180, 45, 15, key="roc_angle")
                angle_rad = np.radians(angle)
                poles = [mag * np.exp(1j*angle_rad), mag * np.exp(-1j*angle_rad)]
            else:
                poles = []
                n_poles = st.number_input("Number of poles:", 2, 5, 3, key="roc_npoles")
                for i in range(n_poles):
                    p = st.number_input(f"Pole {i+1}:", -2.0, 2.0, 0.5*(i+1), key=f"roc_p{i}")
                    poles.append(p)
            
            roc_type = st.radio("ROC Type:", ["Exterior", "Interior", "Annular"], key="roc_type")
        
        with col2:
            st.markdown("### 📊 ROC Visualization")
            
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Draw unit circle
            circle = plt.Circle((0, 0), 1, color='blue', fill=False, linewidth=2, linestyle='--')
            ax.add_patch(circle)
            
            # Draw ROC
            if poles:
                max_pole = max(abs(p) for p in poles)
                min_pole = min(abs(p) for p in poles) if len(poles) > 1 else 0
                
                if roc_type == "Exterior":
                    # ROC outside largest pole
                    ax.fill_between([-2, 2], -2, 2, where=np.ones(2), 
                                   color='green', alpha=0.2)
                    inner_circle = plt.Circle((0, 0), max_pole, color='white', fill=True)
                    ax.add_patch(inner_circle)
                    st.info(f"ROC: |z| > {max_pole:.2f}")
                elif roc_type == "Interior":
                    # ROC inside smallest pole
                    roc_circle = plt.Circle((0, 0), min_pole if min_pole > 0 else max_pole, 
                                           color='green', fill=True, alpha=0.2)
                    ax.add_patch(roc_circle)
                    st.info(f"ROC: |z| < {min_pole if min_pole > 0 else max_pole:.2f}")
                else:  # Annular
                    if len(poles) > 1:
                        outer_circle = plt.Circle((0, 0), max_pole, 
                                                 color='green', fill=True, alpha=0.2)
                        inner_circle = plt.Circle((0, 0), min_pole, 
                                                 color='white', fill=True)
                        ax.add_patch(outer_circle)
                        ax.add_patch(inner_circle)
                        st.info(f"ROC: {min_pole:.2f} < |z| < {max_pole:.2f}")
            
            # Plot poles
            for p in poles:
                if isinstance(p, complex):
                    ax.plot(p.real, p.imag, 'x', markersize=12, 
                           color='red', markeredgewidth=2)
                else:
                    ax.plot(p, 0, 'x', markersize=12, 
                           color='red', markeredgewidth=2)
            
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linewidth=0.5)
            ax.axvline(x=0, color='k', linewidth=0.5)
            ax.set_xlabel('Real', fontweight='bold')
            ax.set_ylabel('Imaginary', fontweight='bold')
            ax.set_title('Region of Convergence', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    elif "Demo 3" in demo_selection:
        st.markdown("## 🔄 Demo 3: Inverse Z-Transform Methods")
        st.markdown("Compare different inverse transform techniques")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Transfer Function")
            
            example = st.selectbox(
                "Select example:",
                ["Simple First Order", "Second Order", "Custom"],
                key="inv_example"
            )
            
            if example == "Simple First Order":
                num = [1]
                den = [1, -0.8]
            elif example == "Second Order":
                num = [1, 0]
                den = [1, -1.5, 0.5]
            else:
                st.markdown("**Numerator coefficients:**")
                num_order = st.number_input("Order:", 0, 3, 1, key="inv_num_ord")
                num = []
                for i in range(num_order + 1):
                    c = st.number_input(f"b{i}:", -10.0, 10.0, 1.0 if i == 0 else 0.0, 
                                       key=f"inv_b{i}")
                    num.append(c)
                
                st.markdown("**Denominator coefficients:**")
                den_order = st.number_input("Order:", 1, 3, 2, key="inv_den_ord")
                den = []
                for i in range(den_order + 1):
                    c = st.number_input(f"a{i}:", -10.0, 10.0, 1.0 if i == 0 else 0.0, 
                                       key=f"inv_a{i}")
                    den.append(c)
            
            n_samples = st.slider("Samples:", 10, 50, 25, key="inv_samples")
        
        with col2:
            st.markdown("### 📈 Comparison of Methods")
            
            # Calculate inverse transforms
            n, y_impulse, y_direct = inverse_z_transform_methods(num, den, n_samples)
            
            # Partial fractions info
            residues, poles, _ = partial_fractions(num, den)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot impulse response
            ax1.stem(n, y_impulse, basefmt=' ', label='Long Division', linefmt='b-')
            ax1.stem(n + 0.1, y_direct, basefmt=' ', label='Partial Fractions', linefmt='r-')
            ax1.set_xlabel('n', fontweight='bold')
            ax1.set_ylabel('x[n]', fontweight='bold')
            ax1.set_title('Inverse Z-Transform Results', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot poles
            ax2.scatter([p.real for p in poles], [p.imag for p in poles], 
                       s=100, c='red', marker='x', linewidths=2)
            circle = plt.Circle((0, 0), 1, color='blue', fill=False, linewidth=2)
            ax2.add_patch(circle)
            ax2.set_xlim(-1.5, 1.5)
            ax2.set_ylim(-1.5, 1.5)
            ax2.set_aspect('equal')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlabel('Real', fontweight='bold')
            ax2.set_ylabel('Imaginary', fontweight='bold')
            ax2.set_title('Pole Locations', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display partial fractions
            st.markdown("#### Partial Fraction Decomposition:")
            for i, (r, p) in enumerate(zip(residues, poles)):
                st.latex(f"\\frac{{{r:.3f}}}{{z - {p:.3f}}}")

# Week 5 Demos
elif "Week 5" in week_selection:
    
    if "Demo 1" in demo_selection:
        st.markdown("## 📊 Demo 1: Pulse Transfer Function Calculator")
        st.markdown("Calculate PTF from continuous system with ZOH")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ System Parameters")
            
            T = st.slider("Sampling Period (s):", 0.1, 2.0, 0.5, 0.1, key="ptf_T")
            
            example = st.selectbox(
                "System type:",
                ["First Order", "Second Order", "Integrator", "Custom"],
                key="ptf_sys"
            )
            
            if example == "First Order":
                K = st.slider("Gain K:", 0.1, 10.0, 1.0, 0.1, key="ptf_K1")
                tau = st.slider("Time constant τ:", 0.1, 5.0, 1.0, 0.1, key="ptf_tau")
                num_s = [K]
                den_s = [tau, 1]
            elif example == "Second Order":
                K = st.slider("Gain K:", 0.1, 10.0, 1.0, 0.1, key="ptf_K2")
                wn = st.slider("ωn:", 0.1, 10.0, 2.0, 0.1, key="ptf_wn")
                zeta = st.slider("ζ:", 0.1, 2.0, 0.7, 0.1, key="ptf_zeta")
                num_s = [K * wn**2]
                den_s = [1, 2*zeta*wn, wn**2]
            elif example == "Integrator":
                K = st.slider("Gain K:", 0.1, 10.0, 1.0, 0.1, key="ptf_Ki")
                num_s = [K]
                den_s = [1, 0]
            else:
                st.markdown("Custom G(s) = b₀/(s² + a₁s + a₀)")
                b0 = st.number_input("b₀:", 0.1, 10.0, 1.0, key="ptf_b0")
                a1 = st.number_input("a₁:", 0.0, 10.0, 1.0, key="ptf_a1")
                a0 = st.number_input("a₀:", 0.0, 10.0, 1.0, key="ptf_a0")
                num_s = [b0]
                den_s = [1, a1, a0]
        
        with col2:
            st.markdown("### 📊 Transfer Functions")
            
            # Continuous system
            G_s = signal.TransferFunction(num_s, den_s)
            
            # Discretize with ZOH
            G_z = signal.cont2discrete((num_s, den_s), T, method='zoh')
            
            # Display
            st.markdown("#### Continuous System G(s):")
            num_str = " + ".join([f"{c:.2f}s^{i}" for i, c in enumerate(num_s[::-1])])
            den_str = " + ".join([f"{c:.2f}s^{i}" for i, c in enumerate(den_s[::-1])])
            st.latex(f"G(s) = \\frac{{{num_str}}}{{{den_str}}}")
            
            st.markdown("#### Discrete System G(z):")
            num_z_str = " + ".join([f"{c:.3f}z^{{{-i}}}" for i, c in enumerate(G_z[0].flatten())])
            den_z_str = " + ".join([f"{c:.3f}z^{{{-i}}}" for i, c in enumerate(G_z[1])])
            st.latex(f"G(z) = \\frac{{{num_z_str}}}{{{den_z_str}}}")
            
            # Step response comparison
            t_cont = np.linspace(0, 10*T, 200)
            t_disc = np.arange(0, 10*T, T)
            
            _, y_cont = signal.step(G_s, T=t_cont)
            _, y_disc = signal.dstep(G_z, n=len(t_disc))
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(t_cont, y_cont, 'b-', label='Continuous', linewidth=2)
            ax.step(t_disc, y_disc[0].flatten(), 'r-', where='post', 
                   label='Discrete (ZOH)', linewidth=2)
            ax.plot(t_disc, y_disc[0].flatten(), 'ro', markersize=8)
            ax.set_xlabel('Time (s)', fontweight='bold')
            ax.set_ylabel('Response', fontweight='bold')
            ax.set_title('Step Response Comparison', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    elif "Demo 2" in demo_selection:
        st.markdown("## 🎯 Demo 2: Block Diagram Analyzer")
        st.markdown("Reduce and analyze block diagram configurations")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Configuration")
            
            config = st.selectbox(
                "Configuration:",
                ["Series", "Parallel", "Feedback"],
                key="bd_config"
            )
            
            st.markdown("**G₁(z) Parameters:**")
            g1_num = st.text_input("Numerator:", "1", key="bd_g1_num")
            g1_den = st.text_input("Denominator:", "1 -0.8", key="bd_g1_den")
            
            st.markdown("**G₂(z) Parameters:**")
            g2_num = st.text_input("Numerator:", "0.5", key="bd_g2_num")
            g2_den = st.text_input("Denominator:", "1 -0.5", key="bd_g2_den")
            
            # Parse coefficients
            try:
                g1_num_coeffs = [float(x) for x in g1_num.split()]
                g1_den_coeffs = [float(x) for x in g1_den.split()]
                g2_num_coeffs = [float(x) for x in g2_num.split()]
                g2_den_coeffs = [float(x) for x in g2_den.split()]
            except:
                st.error("Invalid coefficient format")
                g1_num_coeffs = [1]
                g1_den_coeffs = [1, -0.8]
                g2_num_coeffs = [0.5]
                g2_den_coeffs = [1, -0.5]
        
        with col2:
            st.markdown("### 📊 Block Diagram Analysis")
            
            G1 = signal.TransferFunction(g1_num_coeffs, g1_den_coeffs, dt=True)
            G2 = signal.TransferFunction(g2_num_coeffs, g2_den_coeffs, dt=True)
            
            # Calculate equivalent transfer function
            G_eq = block_diagram_reduction(G1, G2, config.lower())
            
            # Display block diagram
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Block diagram visualization
            ax1.text(0.1, 0.5, 'R(z)', fontsize=12, ha='center')
            
            if config == "Series":
                ax1.add_patch(plt.Rectangle((0.2, 0.4), 0.2, 0.2, 
                                           fill=True, facecolor='lightblue'))
                ax1.text(0.3, 0.5, 'G₁(z)', fontsize=12, ha='center')
                ax1.add_patch(plt.Rectangle((0.5, 0.4), 0.2, 0.2, 
                                           fill=True, facecolor='lightgreen'))
                ax1.text(0.6, 0.5, 'G₂(z)', fontsize=12, ha='center')
                ax1.arrow(0.15, 0.5, 0.04, 0, head_width=0.03, color='black')
                ax1.arrow(0.42, 0.5, 0.07, 0, head_width=0.03, color='black')
                ax1.arrow(0.72, 0.5, 0.08, 0, head_width=0.03, color='black')
                ax1.text(0.85, 0.5, 'Y(z)', fontsize=12, ha='center')
                
            elif config == "Parallel":
                ax1.add_patch(plt.Rectangle((0.3, 0.6), 0.2, 0.15, 
                                           fill=True, facecolor='lightblue'))
                ax1.text(0.4, 0.675, 'G₁(z)', fontsize=12, ha='center')
                ax1.add_patch(plt.Rectangle((0.3, 0.25), 0.2, 0.15, 
                                           fill=True, facecolor='lightgreen'))
                ax1.text(0.4, 0.325, 'G₂(z)', fontsize=12, ha='center')
                ax1.plot([0.2, 0.2], [0.325, 0.675], 'k-')
                ax1.plot([0.6, 0.6], [0.325, 0.675], 'k-')
                ax1.text(0.7, 0.5, '+', fontsize=16, ha='center')
                ax1.text(0.85, 0.5, 'Y(z)', fontsize=12, ha='center')
                
            else:  # Feedback
                ax1.add_patch(plt.Rectangle((0.3, 0.5), 0.2, 0.15, 
                                           fill=True, facecolor='lightblue'))
                ax1.text(0.4, 0.575, 'G₁(z)', fontsize=12, ha='center')
                ax1.add_patch(plt.Rectangle((0.4, 0.2), 0.2, 0.15, 
                                           fill=True, facecolor='lightgreen'))
                ax1.text(0.5, 0.275, 'G₂(z)', fontsize=12, ha='center')
                ax1.text(0.2, 0.575, '⊕', fontsize=16, ha='center')
                ax1.plot([0.7, 0.7, 0.2], [0.575, 0.275, 0.275], 'k-')
                ax1.text(0.85, 0.575, 'Y(z)', fontsize=12, ha='center')
            
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.axis('off')
            ax1.set_title(f'{config} Configuration', fontweight='bold')
            
            # Pole-zero plot
            poles_eq = np.roots(G_eq.den)
            zeros_eq = np.roots(G_eq.num)
            
            draw_unit_circle(ax2, "Equivalent System Poles & Zeros")
            ax2.plot(poles_eq.real, poles_eq.imag, 'x', markersize=12, 
                    color='red', markeredgewidth=2, label='Poles')
            if len(zeros_eq) > 0:
                ax2.plot(zeros_eq.real, zeros_eq.imag, 'o', markersize=8, 
                        color='blue', label='Zeros')
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display equivalent transfer function
            st.markdown("#### Equivalent Transfer Function:")
            num_str = " + ".join([f"{c:.3f}z^{{{len(G_eq.num)-1-i}}}" 
                                 for i, c in enumerate(G_eq.num)])
            den_str = " + ".join([f"{c:.3f}z^{{{len(G_eq.den)-1-i}}}" 
                                 for i, c in enumerate(G_eq.den)])
            st.latex(f"G_{{eq}}(z) = \\frac{{{num_str}}}{{{den_str}}}")

# Week 6 Demos (Enhanced)
elif "Week 6" in week_selection:
    
    if "Demo 1" in demo_selection:
        st.markdown("## 📊 Demo 1: Unit Circle & Stability Visualization")
        st.markdown("""
        <div class="info-box">
            <h4>📚 Stability Criterion (Source: Chakrabortty Ch.7, p.230)</h4>
            <p><strong>Stable:</strong> All poles inside unit circle (|z| < 1)</p>
            <p><strong>Marginally Stable:</strong> Simple poles on unit circle (|z| = 1)</p>
            <p><strong>Unstable:</strong> Any pole outside unit circle (|z| > 1)</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ System Configuration")
            
            example = st.selectbox(
                "Select example:",
                ["Stable System", "Marginally Stable", "Unstable System", "Custom"],
                key="uc_example"
            )
            
            if example == "Stable System":
                poles = [0.5, 0.8*np.exp(1j*np.pi/4), 0.8*np.exp(-1j*np.pi/4)]
            elif example == "Marginally Stable":
                poles = [0.5, np.exp(1j*np.pi/3), np.exp(-1j*np.pi/3)]
            elif example == "Unstable System":
                poles = [0.5, 1.2, 0.9*np.exp(1j*np.pi/6), 0.9*np.exp(-1j*np.pi/6)]
            else:
                n_poles = st.number_input("Number of poles:", 1, 5, 3, key="uc_npoles")
                poles = []
                for i in range(n_poles):
                    col_r, col_i = st.columns(2)
                    with col_r:
                        real = st.number_input(f"Pole {i+1} real:", 
                                             -2.0, 2.0, 0.5, key=f"uc_r{i}")
                    with col_i:
                        imag = st.number_input(f"Pole {i+1} imag:", 
                                             -2.0, 2.0, 0.0, key=f"uc_i{i}")
                    poles.append(real + 1j*imag if imag != 0 else real)
        
        with col2:
            st.markdown("### 📊 Stability Analysis")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Unit circle plot
            draw_unit_circle(ax1, "Pole Locations")
            
            # Color code poles by stability
            for pole in poles:
                mag = abs(pole)
                if mag < 0.99:
                    color = 'green'
                    marker = 'o'
                elif mag < 1.01:
                    color = 'orange'
                    marker = 's'
                else:
                    color = 'red'
                    marker = '^'
                
                if isinstance(pole, complex):
                    ax1.plot(pole.real, pole.imag, marker=marker, 
                            markersize=12, color=color, markeredgewidth=2)
                else:
                    ax1.plot(pole, 0, marker=marker, 
                            markersize=12, color=color, markeredgewidth=2)
            
            # Magnitude plot
            pole_mags = [abs(p) for p in poles]
            pole_indices = range(len(poles))
            colors = ['green' if m < 0.99 else 'orange' if m < 1.01 else 'red' 
                     for m in pole_mags]
            
            ax2.bar(pole_indices, pole_mags, color=colors, alpha=0.7)
            ax2.axhline(y=1, color='blue', linestyle='--', linewidth=2, 
                       label='Unit Circle')
            ax2.set_xlabel('Pole Index', fontweight='bold')
            ax2.set_ylabel('|Pole|', fontweight='bold')
            ax2.set_title('Pole Magnitudes', fontsize=14, fontweight='bold')
            ax2.set_ylim(0, max(pole_mags + [1.2]))
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Stability assessment
            all_stable = all(abs(p) < 0.99 for p in poles)
            any_unstable = any(abs(p) > 1.01 for p in poles)
            
            if all_stable:
                st.success("✅ **STABLE SYSTEM** - All poles inside unit circle")
            elif any_unstable:
                st.error("❌ **UNSTABLE SYSTEM** - At least one pole outside unit circle")
            else:
                st.warning("⚠️ **MARGINALLY STABLE** - Poles on unit circle boundary")
            
            # Detailed pole information
            st.markdown("#### Pole Details:")
            pole_df = pd.DataFrame({
                'Pole': [f'p{i+1}' for i in range(len(poles))],
                'Value': [f'{p:.3f}' if not isinstance(p, complex) 
                         else f'{p.real:.3f}+{p.imag:.3f}j' for p in poles],
                'Magnitude': [f'{abs(p):.3f}' for p in poles],
                'Status': ['Stable' if abs(p) < 0.99 else 
                          'Marginal' if abs(p) < 1.01 else 
                          'Unstable' for p in poles]
            })
            st.dataframe(pole_df, use_container_width=True)
    
    elif "Demo 2" in demo_selection:
        st.markdown("## 🎯 Demo 2: Jury Test Step-by-Step")
        st.markdown("""
        <div class="info-box">
            <h4>📚 Jury Stability Test (Source: DigitalControlTextBook.pdf, pp.104-109)</h4>
            <p>Systematic method to check stability without finding roots</p>
            <ol>
                <li>Check Q(1) > 0</li>
                <li>Check (-1)^n Q(-1) > 0</li>
                <li>Check |a₀| < aₙ</li>
                <li>Construct Jury table and check constraints</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Polynomial Input")
            
            example = st.selectbox(
                "Select example:",
                ["Example 1 (Stable)", "Example 2 (Unstable)", 
                 "Example 3 (Marginal)", "Custom"],
                key="jury_example"
            )
            
            if example == "Example 1 (Stable)":
                coeffs = [0.08, -0.6, 1.0]  # z² - 0.6z + 0.08
            elif example == "Example 2 (Unstable)":
                coeffs = [0.5, -1.5, 1.0]  # z² - 1.5z + 0.5
            elif example == "Example 3 (Marginal)":
                coeffs = [-0.25, -0.5, 1.0]  # z² - 0.5z - 0.25
            else:
                order = st.slider("Polynomial order:", 2, 4, 2, key="jury_order")
                coeffs = []
                for i in range(order + 1):
                    c = st.number_input(f"a{i} (z^{i}):", -10.0, 10.0, 
                                       1.0 if i == order else 0.0, 
                                       key=f"jury_c{i}")
                    coeffs.append(c)
        
        with col2:
            st.markdown("### 📊 Jury Test Analysis")
            
            if st.button("🚀 Run Jury Test", type="primary"):
                is_stable, details = jury_test(coeffs)
                
                # Display polynomial
                poly_str = " + ".join([f"({c:.3f})z^{i}" for i, c in enumerate(coeffs)])
                st.latex(f"Q(z) = {poly_str}")
                
                # Display conditions
                st.markdown("#### Stability Conditions:")
                for cond in details['conditions']:
                    if cond['pass']:
                        st.success(f"✅ Condition {cond['number']}: {cond['description']} "
                                 f"= {cond['value']:.4f if isinstance(cond['value'], float) else cond['value']}")
                    else:
                        st.error(f"❌ Condition {cond['number']}: {cond['description']} "
                               f"= {cond['value']:.4f if isinstance(cond['value'], float) else cond['value']} FAILED")
                        break
                
                # Display Jury table
                if len(details['jury_table']) > 0:
                    st.markdown("#### Jury Table:")
                    
                    # Format table for display
                    max_len = max(len(row) for row in details['jury_table'])
                    table_data = []
                    row_labels = []
                    
                    for i, row in enumerate(details['jury_table']):
                        padded_row = list(row) + [np.nan] * (max_len - len(row))
                        table_data.append(padded_row)
                        if i < 2:
                            row_labels.append(f"Row {i+1}")
                        else:
                            row_labels.append(f"Row {i+1} (b{(i-1)//2})")
                    
                    df = pd.DataFrame(table_data, index=row_labels)
                    # Format dataframe - use map instead of applymap for newer pandas
                    formatted_df = df.copy()
                    for col in formatted_df.columns:
                        formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.6f}" if not np.isnan(x) else "")
                    st.dataframe(formatted_df, use_container_width=True)
                
                # Final result
                if is_stable:
                    st.markdown("""
                    <div class="success-box">
                        <h3>✅ SYSTEM IS STABLE</h3>
                        <p>All Jury test conditions are satisfied.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="danger-box">
                        <h3>❌ SYSTEM IS UNSTABLE</h3>
                        <p>Failed at condition {details['failed_at']}.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Verify with roots
                roots = np.roots(coeffs[::-1])
                st.markdown("#### Verification (Root Finding):")
                roots_df = pd.DataFrame({
                    'Root': [f'r{i+1}' for i in range(len(roots))],
                    'Value': [f'{r:.3f}' if not isinstance(r, complex) 
                             else f'{r.real:.3f}+{r.imag:.3f}j' for r in roots],
                    'Magnitude': [f'{abs(r):.3f}' for r in roots],
                    'Inside Unit Circle': [abs(r) < 1 for r in roots]
                })
                st.dataframe(roots_df, use_container_width=True)
    
    elif "Demo 3" in demo_selection:
        st.markdown("## 🔄 Demo 3: Parametric Stability Analysis (K Range)")
        st.markdown("""
        <div class="info-box">
            <h4>📚 Finding Stable Gain Range (Source: Chakrabortty Example 7.4)</h4>
            <p>Determine range of controller gain K for stability</p>
            <p>Characteristic equation: Q(z,K) = 0</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ System Parameters")
            
            example = st.selectbox(
                "Select system:",
                ["Example 7.4 (Chakrabortty)", "Simple Second Order", "Custom"],
                key="k_example"
            )
            
            if example == "Example 7.4 (Chakrabortty)":
                # z² + (0.368K - 1.368)z + (0.368 + 0.264K) = 0
                st.markdown("Q(z) = z² + (0.368K - 1.368)z + (0.368 + 0.264K)")
                a2 = 1.0
                a1_K = 0.368
                a1_const = -1.368
                a0_K = 0.264
                a0_const = 0.368
            elif example == "Simple Second Order":
                st.markdown("Q(z) = z² - z + K")
                a2 = 1.0
                a1_K = 0.0
                a1_const = -1.0
                a0_K = 1.0
                a0_const = 0.0
            else:
                st.markdown("Q(z) = a₂z² + (b₁K + c₁)z + (b₀K + c₀)")
                a2 = st.number_input("a₂:", 0.1, 10.0, 1.0, key="k_a2")
                a1_K = st.number_input("b₁:", -10.0, 10.0, 0.0, key="k_b1")
                a1_const = st.number_input("c₁:", -10.0, 10.0, -1.0, key="k_c1")
                a0_K = st.number_input("b₀:", -10.0, 10.0, 1.0, key="k_b0")
                a0_const = st.number_input("c₀:", -10.0, 10.0, 0.0, key="k_c0")
            
            K_test = st.slider("Test K value:", 0.0, 10.0, 1.0, 0.1, key="k_test")
            K_range = np.linspace(0, 10, 200)
        
        with col2:
            st.markdown("### 📊 Stability Range Analysis")
            
            if st.button("🚀 Find Stable Range", type="primary"):
                
                # Find stable range
                stable_K = []
                K_min, K_max = None, None
                
                for K in K_range:
                    coeffs = [a0_const + a0_K*K, a1_const + a1_K*K, a2]
                    is_stable, _ = jury_test(coeffs)
                    if is_stable:
                        stable_K.append(K)
                        if K_min is None:
                            K_min = K
                        K_max = K
                
                if stable_K:
                    st.success(f"✅ Stable range: {K_min:.3f} < K < {K_max:.3f}")
                    
                    # Visualization
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                    
                    # Stability region plot
                    ax1.fill_between(K_range, 0, 1, 
                                    where=[K_min <= k <= K_max for k in K_range],
                                    color='green', alpha=0.3, label='Stable Region')
                    ax1.fill_between(K_range, 0, 1, 
                                    where=[(k < K_min or k > K_max) for k in K_range],
                                    color='red', alpha=0.3, label='Unstable Region')
                    
                    ax1.axvline(K_test, color='blue', linewidth=2.5, 
                              label=f'Test K={K_test:.2f}')
                    ax1.axvline(K_min, color='green', linestyle='--', linewidth=2)
                    ax1.axvline(K_max, color='green', linestyle='--', linewidth=2)
                    
                    ax1.set_xlabel('Gain K', fontsize=12, fontweight='bold')
                    ax1.set_ylabel('Stability', fontsize=12, fontweight='bold')
                    ax1.set_title('Stability vs Gain K', fontsize=14, fontweight='bold')
                    ax1.set_ylim(0, 1)
                    ax1.set_yticks([])
                    ax1.legend(fontsize=10)
                    ax1.grid(True, alpha=0.3, axis='x')
                    
                    # Root locus for test K
                    test_coeffs = [a0_const + a0_K*K_test, a1_const + a1_K*K_test, a2]
                    roots = np.roots(test_coeffs[::-1])
                    
                    draw_unit_circle(ax2, f"Poles at K={K_test:.2f}")
                    for root in roots:
                        color = 'green' if abs(root) < 1 else 'red'
                        if isinstance(root, complex):
                            ax2.plot(root.real, root.imag, 'o', markersize=12, 
                                   color=color, markeredgewidth=2)
                        else:
                            ax2.plot(root, 0, 'o', markersize=12, 
                                   color=color, markeredgewidth=2)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Test specific K value
                    test_stable, test_details = jury_test(test_coeffs)
                    if test_stable:
                        st.success(f"✅ K = {K_test} is STABLE")
                    else:
                        st.error(f"❌ K = {K_test} is UNSTABLE")
                else:
                    st.error("❌ No stable range found in K ∈ [0, 10]")
    
    elif "Demo 4" in demo_selection:
        st.markdown("## 📐 Demo 4: Bilinear Transformation")
        st.markdown("""
        <div class="info-box">
            <h4>📚 Bilinear Transform (Source: Chakrabortty Ch.7, pp.234-236)</h4>
            <p><strong>w = (z-1)/(z+1)</strong> or <strong>z = (1+w)/(1-w)</strong></p>
            <p>Maps: Inside unit circle ↔ Left-half plane</p>
            <p>Enables use of Routh-Hurwitz criterion for discrete systems</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Polynomial in z")
            
            order = st.selectbox("Polynomial order:", [2, 3, 4], index=0, key="bt_order")
            
            coeffs_z = []
            for i in range(order + 1):
                val = st.number_input(f"a{i} (z^{i}):", -10.0, 10.0, 
                                     1.0 if i == order else 
                                     (-0.6 if i == 1 else 0.08 if i == 0 else 0.0), 
                                     format="%.4f", key=f"bt_c{i}")
                coeffs_z.append(val)
        
        with col2:
            st.markdown("### 🔄 Transformation Results")
            
            if st.button("🚀 Apply Transform", type="primary"):
                try:
                    w_coeffs = bilinear_transform(coeffs_z)
                    
                    # Display polynomials
                    poly_z_str = " + ".join([f"({c:.3f})z^{i}" for i, c in enumerate(coeffs_z)])
                    st.markdown("**Q(z):**")
                    st.latex(f"Q(z) = {poly_z_str}")
                    
                    st.markdown("**Q(w) after transformation:**")
                    poly_w_str = " + ".join([f"({c:.3f})w^{i}" for i, c in enumerate(w_coeffs)])
                    st.latex(f"Q(w) = {poly_w_str}")
                    
                    # Calculate roots
                    roots_z = np.roots(coeffs_z[::-1])
                    roots_w = np.roots(w_coeffs[::-1])
                    
                    # Visualization
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                    
                    # Z-plane
                    draw_unit_circle(ax1, "Z-Plane")
                    for root in roots_z:
                        mag = abs(root)
                        color = 'green' if mag < 1 else 'red' if mag > 1 else 'orange'
                        if isinstance(root, complex):
                            ax1.plot(root.real, root.imag, 'o', color=color, 
                                   markersize=12, markeredgewidth=2)
                        else:
                            ax1.plot(root, 0, 'o', color=color, 
                                   markersize=12, markeredgewidth=2)
                    
                    # W-plane
                    ax2.axvline(x=0, color='blue', linewidth=2.5, label='Stability Boundary')
                    ax2.fill_betweenx([-3, 3], -3, 0, color='green', alpha=0.2, label='Stable (LHP)')
                    ax2.fill_betweenx([-3, 3], 0, 3, color='red', alpha=0.2, label='Unstable (RHP)')
                    
                    for root in roots_w:
                        color = 'green' if root.real < 0 else 'red' if root.real > 0 else 'orange'
                        if isinstance(root, complex):
                            ax2.plot(root.real, root.imag, 'o', color=color, 
                                   markersize=12, markeredgewidth=2)
                        else:
                            ax2.plot(root.real, 0, 'o', color=color, 
                                   markersize=12, markeredgewidth=2)
                    
                    ax2.set_xlabel('Real (σ)', fontsize=12, fontweight='bold')
                    ax2.set_ylabel('Imaginary (jω)', fontsize=12, fontweight='bold')
                    ax2.set_title('W-Plane', fontsize=14, fontweight='bold')
                    ax2.grid(True, alpha=0.3)
                    ax2.legend(fontsize=10, loc='upper right')
                    ax2.set_xlim(-3, 3)
                    ax2.set_ylim(-3, 3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Stability assessment
                    stable_z = all(abs(r) < 1 for r in roots_z)
                    stable_w = all(r.real < 0 for r in roots_w)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Z-plane Analysis:**")
                        st.markdown(f"- All poles inside unit circle: {'✅ Yes' if stable_z else '❌ No'}")
                        st.markdown(f"- **Result: {'STABLE' if stable_z else 'UNSTABLE'}**")
                    
                    with col2:
                        st.markdown("**W-plane Analysis:**")
                        st.markdown(f"- All poles in LHP: {'✅ Yes' if stable_w else '❌ No'}")
                        st.markdown(f"- **Result: {'STABLE' if stable_w else 'UNSTABLE'}**")
                    
                    if stable_z == stable_w:
                        st.success("✅ Both analyses agree! Transformation preserved stability.")
                    else:
                        st.warning("⚠️ Mismatch detected - check for numerical errors.")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    elif "Demo 5" in demo_selection:
        st.markdown("## 🔧 Demo 5: Routh Criterion in w-plane")
        st.markdown("""
        <div class="info-box">
            <h4>📚 Routh Criterion (Source: Chakrabortty Ch.7, pp.236-239)</h4>
            <p><strong>Procedure:</strong></p>
            <ol>
                <li>Transform Q(z) → Q(w) using bilinear transformation</li>
                <li>Construct Routh array from Q(w)</li>
                <li>Check first column for sign changes</li>
                <li>No sign changes → Stable</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Input Polynomial")
            
            example = st.selectbox(
                "Choose example:",
                ["Example 1: Stable", "Example 2: Unstable", 
                 "Parametric (K=1.2)", "Custom"],
                key="routh_example"
            )
            
            if example == "Example 1: Stable":
                coeffs_z = [0.08, -0.6, 1.0]
            elif example == "Example 2: Unstable":
                coeffs_z = [0.5, -1.5, 1.0]
            elif example == "Parametric (K=1.2)":
                K = 1.2
                # Ensure float conversion
                coeffs_z = [float(0.368 + 0.264*K), float(0.368*K - 1.368), 1.0]
            else:
                order = st.slider("Polynomial order:", 2, 4, 2, key="routh_order")
                coeffs_z = []
                for i in range(order + 1):
                    c = st.number_input(f"a{i} (z^{i}):", -10.0, 10.0, 
                                       1.0 if i == order else 0.0, 
                                       key=f"routh_c{i}")
                    coeffs_z.append(float(c))
        
        with col2:
            st.markdown("### 📊 Routh Analysis")
            
            if st.button("🚀 Analyze with Routh", type="primary"):
                # Step 1: Bilinear transformation
                st.markdown("#### Step 1: Bilinear Transformation")
                
                try:
                    # Ensure coefficients are numpy array of floats
                    coeffs_z_array = np.array(coeffs_z, dtype=float)
                    
                    # Bilinear transformation
                    try:
                        coeffs_w = bilinear_transform(coeffs_z_array)
                    except Exception as e:
                        st.error(f"Error in bilinear transformation: {str(e)}")
                        raise
                    
                    # Display polynomials
                    try:
                        poly_z_str = " + ".join([f"({c:.3f})z^{i}" for i, c in enumerate(coeffs_z_array)])
                        st.latex(f"Q(z) = {poly_z_str}")
                        
                        poly_w_str = " + ".join([f"({c:.4f})w^{i}" for i, c in enumerate(coeffs_w)])
                        st.latex(f"Q_w(w) = {poly_w_str}")
                    except Exception as e:
                        st.error(f"Error displaying polynomials: {str(e)}")
                        raise
                    
                    # Step 2: Construct Routh array
                    st.markdown("#### Step 2: Routh Array")
                    
                    routh = routh_array(coeffs_w)
                    
                    # Display Routh array
                    n = len(coeffs_w) - 1
                    row_labels = [f"w^{n-i}" for i in range(n+1)]
                    
                    df = pd.DataFrame(routh, index=row_labels)
                    df.columns = [f"Col {i+1}" for i in range(routh.shape[1])]
                    
                    # Highlight first column
                    def highlight_first_col(row):
                        return ['background-color: #ffffcc' if i == 0 else '' 
                               for i in range(len(row))]
                    
                    styled_df = df.style.apply(highlight_first_col, axis=1)\
                                       .format("{:.6f}")\
                                       .set_properties(**{'text-align': 'center'})
                    
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Step 3: Check stability
                    st.markdown("#### Step 3: Stability Check")
                    
                    is_stable, sign_changes = check_routh_stability(routh)
                    
                    st.markdown("**First Column Analysis:**")
                    first_col = routh[:, 0]
                    first_col = first_col[first_col != 0]
                    
                    for i, val in enumerate(first_col):
                        st.markdown(f"- Row {i+1}: {val:.6f}")
                    
                    st.markdown(f"\n**Sign changes in first column:** {sign_changes}")
                    
                    if is_stable:
                        st.markdown("""
                        <div class='success-box'>
                        <h3>✅ SYSTEM IS STABLE</h3>
                        <p>No sign changes in first column of Routh array.</p>
                        <p>All poles are in left-half w-plane (inside unit circle in z-plane).</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='danger-box'>
                        <h3>❌ SYSTEM IS UNSTABLE</h3>
                        <p>Number of sign changes: {sign_changes}</p>
                        <p>This indicates {sign_changes} pole(s) in right-half w-plane.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Verification
                    st.markdown("#### 🔬 Verification")
                    
                    roots_z = np.roots(coeffs_z[::-1])
                    all_inside = all(abs(r) < 1 for r in roots_z)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Routh Criterion:**")
                        st.markdown(f"- {'STABLE ✅' if is_stable else 'UNSTABLE ❌'}")
                        st.markdown(f"- Sign changes: {sign_changes}")
                    
                    with col2:
                        st.markdown("**Direct Root Finding:**")
                        st.markdown(f"- {'STABLE ✅' if all_inside else 'UNSTABLE ❌'}")
                        st.markdown(f"- Roots:")
                        for i, root in enumerate(roots_z):
                            st.markdown(f"  z{i+1}: |z| = {abs(root):.4f}")
                    
                    if is_stable == all_inside:
                        st.success("✅ Routh criterion and root-finding agree!")
                    else:
                        st.warning("⚠️ Results differ. Check for numerical errors.")
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.info("Try a different polynomial or check coefficient values.")

# ═══════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px;'>
        <p><strong>📚 Digital Control Systems - Weeks 4, 5 & 6</strong></p>
        <p>Dr. Ali Al-Ghanimi | Electrical Engineering | University of Kufa</p>
        <p>📧 ali.alghanimi@uokufa.edu.iq | 🏢 EE Building, Room 115</p>
        <p style='font-size: 0.9rem; color: #666; margin-top: 1rem;'>
            Based on: Chakrabortty et al. "Digital Control System Analysis & Design", 
            DigitalControlTextBook.pdf, DCS.pdf
        </p>
        <p style='font-size: 0.9rem; color: #666;'>
            © 2025 University of Kufa | All rights reserved
        </p>
    </div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# END OF INTEGRATED DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════
