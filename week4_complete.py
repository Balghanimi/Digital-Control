# ═══════════════════════════════════════════════════════════════════
# WEEK 6: STABILITY ANALYSIS I - INTERACTIVE DEMONSTRATIONS
# Digital Control Systems | University of Kufa
# Dr. Ali Al-Ghanimi | Electrical Engineering Department
# 
# Topics: Jury Test & Routh-Hurwitz Criterion (via Bilinear Transform)
# ═══════════════════════════════════════════════════════════════════

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from datetime import datetime
import sympy as sp

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Week 6: Stability Analysis I | Dr. Al-Ghanimi",
    page_icon="🎯",
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
    .danger-box {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .info-box {
        background: #d1ecf1;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    .stAlert {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
    <div class="main-header">
        <h1>🎓 Digital Control Systems</h1>
        <h2>Week 6: Stability Analysis I</h2>
        <h3>Jury Test & Routh-Hurwitz Criterion</h3>
        <p style="font-size: 1.1rem; margin-top: 1rem;">
            <strong>Dr. Ali Al-Ghanimi</strong><br>
            Electrical Engineering Department<br>
            University of Kufa | Academic Year 2025
        </p>
    </div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR - DEMO SELECTION
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🎯 Select Demonstration")
    
    demo_selection = st.radio(
        "Choose Demo:",
        ["📊 Demo 1: Unit Circle & Stability Regions",
         "🎯 Demo 2: Jury Test - Step by Step",
         "🔄 Demo 3: Parametric Stability (Find K Range)",
         "📐 Demo 4: Bilinear Transformation",
         "🔧 Demo 5: Routh Criterion in w-plane"],
        key="demo_nav"
    )
    
    st.markdown("---")
    
    st.markdown("### 🎯 Learning Objectives")
    with st.expander("Week 6 Goals"):
        st.markdown("""
        1. ✅ Apply Jury stability test
        2. ✅ Construct Jury table systematically
        3. ✅ Use bilinear transformation
        4. ✅ Apply Routh-Hurwitz criterion
        5. ✅ Find stable parameter ranges
        6. ✅ Compare different methods
        """)
    
    st.markdown("---")
    
    st.markdown("### 📖 Resources")
    st.info("""
    **Main Sources:**
    - Chakrabortty Ch.7 (pp. 230-244)
    - DCS.pdf (Jury & Bilinear)
    - DigitalControlTextBook.pdf Ch.4
    - courssystèmes... (proofs)
    """)
    
    st.markdown("---")
    
    st.markdown("### 💡 Quick Reference")
    with st.expander("Stability Conditions"):
        st.markdown("""
        **z-plane:** Poles inside unit circle
        
        $|z_i| < 1$ for all i → STABLE
        
        **Jury Test:** $(n+1)$ conditions
        
        **Routh:** All first column same sign
        """)
    
    st.markdown("---")
    
    st.markdown("### 💬 Contact")
    st.markdown("""
    📧 ali.alghanimi@uokufa.edu.iq  
    🏢 EE Building, Room 301  
    🕐 Office Hours: Sun-Thu, 10-12 PM
    """)

# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def draw_unit_circle(ax, shade_stable=True):
    """Draw unit circle with optional shading of stable region"""
    theta = np.linspace(0, 2*np.pi, 200)
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)
    
    ax.plot(x_circle, y_circle, 'b-', linewidth=2.5, label='Unit Circle (|z|=1)')
    
    if shade_stable:
        # Shade stable region (inside)
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
    
    Parameters:
    -----------
    coeffs : list
        Polynomial coefficients [a0, a1, ..., an] (ascending powers)
        
    Returns:
    --------
    stable : bool
        True if stable, False otherwise
    conditions : list
        Results of each condition check
    table : list
        Jury table rows
    
    Source: Chakrabortty Ch.7, Section 7.5, pp. 239-242
           DigitalControlTextBook.pdf, Chapter 4.5, pp. 104-109
    """
    
    n = len(coeffs) - 1  # Order of polynomial
    a = np.array(coeffs)
    
    # Ensure an > 0
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
        # Add reversed row
        table.append(current_row[::-1])
        
        # Calculate next row
        next_row = []
        for k in range(len(current_row) - 1):
            element = current_row[0] * current_row[k] - current_row[-1] * current_row[-(k+1)]
            next_row.append(element)
        
        table.append(np.array(next_row))
        
        # Check condition: |first| > |last|
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
    
    Parameters:
    -----------
    z_poly : list
        Coefficients of polynomial in z [a0, a1, ..., an]
        
    Returns:
    --------
    w_poly : list
        Coefficients of polynomial in w
        
    Source: Chakrabortty Ch.7, Section 7.3, pp. 234-236
           DCS.pdf, Bilinear Transformation section
    """
    z = sp.Symbol('z')
    w = sp.Symbol('w')
    
    # Build polynomial in z
    poly_z = sum([z_poly[i] * z**i for i in range(len(z_poly))])
    
    # Substitute z = (w+1)/(w-1)
    poly_w = poly_z.subs(z, (w+1)/(w-1))
    
    # Multiply by (w-1)^n to clear denominator
    n = len(z_poly) - 1
    poly_w = sp.simplify(poly_w * (w-1)**n)
    
    # Extract coefficients
    poly_w = sp.expand(poly_w)
    w_coeffs = [float(poly_w.coeff(w, i)) for i in range(sp.degree(poly_w, w) + 1)]
    
    return w_coeffs, poly_w

def routh_array(coeffs):
    """
    Construct Routh array
    
    Parameters:
    -----------
    coeffs : list
        Polynomial coefficients [a0, a1, ..., an] (ascending powers)
        
    Returns:
    --------
    array : numpy.ndarray
        Routh array
    stable : bool
        True if stable (all first column same sign)
        
    Source: Standard control theory (assumed prerequisite)
    """
    
    # Reverse to get descending powers
    coeffs = coeffs[::-1]
    n = len(coeffs) - 1
    
    # Initialize array
    routh = np.zeros((n + 1, (n + 2) // 2))
    
    # First two rows
    routh[0, :] = coeffs[0::2]
    routh[1, :len(coeffs[1::2])] = coeffs[1::2]
    
    # Calculate remaining rows
    for i in range(2, n + 1):
        for j in range((n + 2) // 2 - 1):
            if routh[i-1, 0] == 0:
                routh[i-1, 0] = 1e-10  # Avoid division by zero
            
            if j < routh.shape[1] - 1:
                routh[i, j] = (routh[i-1, 0] * routh[i-2, j+1] - 
                              routh[i-2, 0] * routh[i-1, j+1]) / routh[i-1, 0]
    
    # Check first column for sign changes
    first_col = routh[:, 0]
    first_col = first_col[first_col != 0]  # Remove zeros
    
    sign_changes = 0
    for i in range(len(first_col) - 1):
        if first_col[i] * first_col[i+1] < 0:
            sign_changes += 1
    
    stable = sign_changes == 0
    
    return routh, stable, first_col

# ═══════════════════════════════════════════════════════════════════
# DEMO 1: UNIT CIRCLE & STABILITY REGIONS
# ═══════════════════════════════════════════════════════════════════

if "Demo 1" in demo_selection:
    st.markdown("## 📊 Demo 1: Unit Circle & Stability Regions")
    st.markdown("Visualize pole locations and determine stability in the z-plane")
    
    st.markdown("""
    <div class="info-box">
        <h4>📚 Theory (Source: Chakrabortty Ch.7, pp. 230-232)</h4>
        <p>For discrete-time systems, stability is determined by pole locations in the z-plane:</p>
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
        
        # Plot poles
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
            
            # Draw line from origin
            ax.plot([0, pole.real], [0, pole.imag], 'k--', alpha=0.3, linewidth=1)
            
            # Annotate with magnitude
            ax.text(pole.real, pole.imag + 0.15, f'|z|={mag:.3f}', 
                   ha='center', fontsize=9, fontweight='bold')
        
        ax.set_title('Pole Locations in Z-Plane', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Analysis
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
                <p><strong>Action:</strong> Redesign controller to move poles inside!</p>
            </div>
            """, unsafe_allow_html=True)
        elif any_marginal:
            st.markdown("""
            <div class="warning-box">
                <h4>⚠️ System is MARGINALLY STABLE</h4>
                <p>Poles on the unit circle cause sustained oscillations.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed analysis
        st.markdown("**Pole Analysis:**")
        pole_data = {
            "Pole": [f"z₁", "z₂", "z₃"],
            "Value": [f"{p.real:.3f} + {p.imag:.3f}j" for p in poles],
            "Magnitude": [f"{abs(p):.4f}" for p in poles],
            "Status": ["✅ Stable" if abs(p) < 1 else 
                      "⚠️ Marginal" if 0.95 < abs(p) < 1.05 else 
                      "❌ Unstable" for p in poles]
        }
        df = pd.DataFrame(pole_data)
        st.table(df)

# ═══════════════════════════════════════════════════════════════════
# DEMO 2: JURY TEST - STEP BY STEP
# ═══════════════════════════════════════════════════════════════════

elif "Demo 2" in demo_selection:
    st.markdown("## 🎯 Demo 2: Jury Test - Step by Step")
    st.markdown("Apply Jury stability test with detailed step-by-step calculations")
    
    st.markdown("""
    <div class="info-box">
        <h4>📚 Jury Test (Source: Chakrabortty Ch.7, pp. 239-242)</h4>
        <p>For characteristic equation Q(z) = aₙzⁿ + ... + a₁z + a₀ = 0:</p>
        <p><strong>(n+1) Conditions must be satisfied:</strong></p>
        <ol>
            <li>Q(1) > 0</li>
            <li>(-1)ⁿ Q(-1) > 0</li>
            <li>|a₀| < aₙ</li>
            <li>Jury table conditions: |bₖ₀| > |bₖ,last| for each row</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Polynomial Coefficients")
        
        order = st.selectbox("Select polynomial order:", [2, 3, 4], index=1)
        
        st.markdown(f"**Enter coefficients for Q(z) = aₙzⁿ + ... + a₀:**")
        
        coeffs = []
        for i in range(order + 1):
            val = st.number_input(f"a_{i} (coefficient of z^{i}):", 
                                 value=0.0 if i < order else 1.0, 
                                 step=0.1, format="%.4f", key=f"coeff_{i}")
            coeffs.append(val)
        
        # Preset examples
        st.markdown("**Or try these examples:**")
        if st.button("Example 1: Stable 2nd order"):
            coeffs = [0.08, -0.6, 1.0]
        if st.button("Example 2: Unstable 3rd order"):
            coeffs = [0.3, 0.07, -1.2, 1.0]
        if st.button("Example 3: Marginal"):
            coeffs = [-0.25, 0.25, -1.0, 1.0]
    
    with col2:
        st.markdown("### 📊 Jury Test Execution")
        
        if st.button("🚀 Run Jury Test", type="primary"):
            
            stable, conditions, table = jury_test(coeffs)
            
            st.markdown("#### Step-by-Step Analysis")
            
            # Display conditions
            for idx, (cond_name, cond_value, cond_result) in enumerate(conditions):
                if cond_result:
                    st.success(f"✅ Condition {idx+1}: {cond_name} → {cond_value} ✓")
                else:
                    st.error(f"❌ Condition {idx+1}: {cond_name} → {cond_value} ✗")
                    if idx < 3:
                        st.stop()
            
            # Display Jury table
            st.markdown("#### Jury Table")
            
            table_display = []
            for i, row in enumerate(table):
                row_label = f"Row {i+1}"
                row_data = [f"{x:.6f}" for x in row]
                table_display.append([row_label] + row_data)
            
            # Create DataFrame for nice display
            max_len = max(len(row) for row in table)
            table_df = pd.DataFrame(table_display, 
                                   columns=["Row"] + [f"Col {j}" for j in range(max_len)])
            st.dataframe(table_df, use_container_width=True)
            
            # Final verdict
            st.markdown("---")
            if stable:
                st.markdown("""
                <div class="success-box">
                    <h3>✅ SYSTEM IS STABLE</h3>
                    <p>All (n+1) Jury conditions are satisfied!</p>
                    <p>All poles lie strictly inside the unit circle.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="danger-box">
                    <h3>❌ SYSTEM IS UNSTABLE</h3>
                    <p>At least one Jury condition failed.</p>
                    <p>One or more poles lie outside or on the unit circle.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Verify by computing actual roots
            st.markdown("#### Verification: Actual Pole Locations")
            roots = np.roots(coeffs[::-1])  # Reverse for numpy convention
            
            fig, ax = plt.subplots(figsize=(8, 8))
            draw_unit_circle(ax)
            
            for i, root in enumerate(roots):
                mag = abs(root)
                if mag < 1:
                    color = 'green'
                elif abs(mag - 1) < 0.01:
                    color = 'orange'
                else:
                    color = 'red'
                
                ax.plot(root.real, root.imag, 'o', color=color, markersize=12,
                       label=f'z_{i+1} = {root:.4f} (|z|={mag:.4f})')
            
            ax.set_title('Computed Pole Locations', fontsize=13, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.info("""
            **Source:** Chakrabortty Ch.7, Example 7.4, pp. 241-242  
            **Methodology:** DigitalControlTextBook.pdf, Section 4.5, pp. 104-109
            """)

# ═══════════════════════════════════════════════════════════════════
# DEMO 3: PARAMETRIC STABILITY (FIND K RANGE)
# ═══════════════════════════════════════════════════════════════════

elif "Demo 3" in demo_selection:
    st.markdown("## 🔄 Demo 3: Parametric Stability Analysis")
    st.markdown("Find the range of gain K for which the system remains stable")
    
    st.markdown("""
    <div class="info-box">
        <h4>📚 Parametric Analysis (Source: Chakrabortty Example 7.4, p. 242)</h4>
        <p>When system has parameter K, apply Jury conditions to find constraints:</p>
        <ul>
            <li>Each condition gives an inequality in K</li>
            <li>Stable range is <strong>intersection</strong> of all constraints</li>
            <li>Critical values occur where system becomes marginally stable</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ System Configuration")
        
        st.markdown("**Closed-Loop Characteristic Equation:**")
        st.latex(r"Q(z) = z^2 + (aK + b)z + (c + dK) = 0")
        
        st.markdown("**Enter coefficients:**")
        a = st.number_input("Coefficient 'a' of K in z¹ term:", value=0.368, format="%.4f")
        b = st.number_input("Constant 'b' in z¹ term:", value=-1.368, format="%.4f")
        c = st.number_input("Constant 'c' in z⁰ term:", value=0.368, format="%.4f")
        d = st.number_input("Coefficient 'd' of K in z⁰ term:", value=0.264, format="%.4f")
        
        st.markdown("**Example: Unity Feedback System**")
        st.markdown("From Chakrabortty Example 7.4:")
        if st.button("Load Example System"):
            a, b, c, d = 0.368, -1.368, 0.368, 0.264
        
        K_test = st.slider("Test gain K:", 0.0, 5.0, 1.2, 0.1)
    
    with col2:
        st.markdown("### 📈 Stability Analysis")
        
        # Symbolic analysis
        st.markdown("#### Condition 1: Q(1) > 0")
        Q_1_expr = f"1 + ({a}K + {b}) + ({c} + {d}K)"
        Q_1_simplified = f"{a + d}K + {1 + b + c}"
        Q_1_coeff = a + d
        Q_1_const = 1 + b + c
        
        if Q_1_coeff > 0:
            K_lower_1 = -Q_1_const / Q_1_coeff
            st.success(f"Q(1) = {Q_1_simplified} > 0")
            st.markdown(f"**Constraint:** K > {K_lower_1:.4f}")
        else:
            st.warning("Special case: coefficient is negative or zero")
        
        # Condition 2
        st.markdown("#### Condition 2: (-1)² Q(-1) > 0")
        Q_minus1_expr = f"1 - ({a}K + {b}) + ({c} + {d}K)"
        Q_minus1_simplified = f"{-a + d}K + {1 - b + c}"
        Q_minus1_coeff = -a + d
        Q_minus1_const = 1 - b + c
        
        if Q_minus1_coeff != 0:
            K_upper_2 = -Q_minus1_const / Q_minus1_coeff if Q_minus1_coeff < 0 else float('inf')
            st.success(f"Q(-1) = {Q_minus1_simplified} > 0")
            if K_upper_2 < float('inf'):
                st.markdown(f"**Constraint:** K < {K_upper_2:.4f}")
        
        # Condition 3
        st.markdown("#### Condition 3: |a₀| < a₂")
        st.markdown(f"|{c} + {d}K| < 1")
        
        if d > 0:
            K_upper_3 = (1 - c) / d
            st.success(f"Assuming K > 0: {c} + {d}K < 1")
            st.markdown(f"**Constraint:** K < {K_upper_3:.4f}")
        
        # Final range
        st.markdown("---")
        st.markdown("### 🎯 Stable Range of K")
        
        K_min = max(0, K_lower_1)
        K_max = min(K_upper_2, K_upper_3)
        
        if K_max > K_min:
            st.markdown(f"""
            <div class="success-box">
                <h3>Stable Range Found:</h3>
                <h2 style="text-align:center; color:green;">{K_min:.4f} < K < {K_max:.4f}</h2>
                <p>System is marginally stable at K = {K_max:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Visualize
            K_range = np.linspace(0, K_max * 1.5, 200)
            stability = [(K >= K_min and K <= K_max) for K in K_range]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Shade regions
            ax.fill_between(K_range, 0, 1, where=[k < K_min or k > K_max for k in K_range], 
                           color='red', alpha=0.3, label='Unstable Region')
            ax.fill_between([K_min, K_max], 0, 1, color='green', alpha=0.3, label='Stable Region')
            
            # Mark boundaries
            ax.axvline(K_min, color='blue', linestyle='--', linewidth=2, label=f'K_min = {K_min:.3f}')
            ax.axvline(K_max, color='red', linestyle='--', linewidth=2, label=f'K_max = {K_max:.3f}')
            
            # Mark test point
            if K_min <= K_test <= K_max:
                ax.axvline(K_test, color='green', linestyle='-', linewidth=2.5, 
                          label=f'Test K = {K_test:.2f} (STABLE)')
            else:
                ax.axvline(K_test, color='darkred', linestyle='-', linewidth=2.5, 
                          label=f'Test K = {K_test:.2f} (UNSTABLE)')
            
            ax.set_xlabel('Gain K', fontsize=13, fontweight='bold')
            ax.set_ylabel('', fontsize=13)
            ax.set_title('Stability Region for Parameter K', fontsize=14, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.legend(fontsize=11, loc='upper right')
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Test current K
            if K_min <= K_test <= K_max:
                st.success(f"✅ K = {K_test} is in stable range!")
            else:
                st.error(f"❌ K = {K_test} is OUTSIDE stable range!")
            
        else:
            st.error("❌ No stable range exists for this system!")
        
        st.info("""
        **Source:** Chakrabortty Ch.7, Example 7.4, pp. 241-242  
        **System:** Same as Example 6.4 (ZOH with plant)
        """)

# ═══════════════════════════════════════════════════════════════════
# DEMO 4: BILINEAR TRANSFORMATION
# ═══════════════════════════════════════════════════════════════════

elif "Demo 4" in demo_selection:
    st.markdown("## 📐 Demo 4: Bilinear Transformation")
    st.markdown("Transform from z-plane to w-plane for stability analysis")
    
    st.markdown("""
    <div class="info-box">
        <h4>📚 Bilinear Transform (Source: Chakrabortty Ch.7, pp. 234-236)</h4>
        <p><strong>Transformation:</strong> w = (z+1)/(z-1) or z = (w+1)/(w-1)</p>
        <p><strong>Key Property:</strong></p>
        <ul>
            <li>Inside unit circle (z-plane) ↔ Left-half plane (w-plane)</li>
            <li>Outside unit circle ↔ Right-half plane</li>
            <li>On unit circle ↔ Imaginary axis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Input Polynomial in z")
        
        order = st.selectbox("Polynomial order:", [2, 3], index=0, key="bt_order")
        
        st.markdown(f"**Q(z) = a₂z² + a₁z + a₀ = 0**" if order == 2 else 
                   "**Q(z) = a₃z³ + a₂z² + a₁z + a₀ = 0**")
        
        coeffs_z = []
        for i in range(order + 1):
            val = st.number_input(f"a_{i}:", value=1.0 if i == order else 
                                 (-0.6 if i == 1 else 0.08), 
                                 format="%.4f", key=f"bt_coeff_{i}")
            coeffs_z.append(val)
        
        if st.button("Example: z² - 0.6z + 0.08"):
            coeffs_z = [0.08, -0.6, 1.0]
    
    with col2:
        st.markdown("### 🔄 Transformation Process")
        
        if st.button("🚀 Apply Bilinear Transform", type="primary"):
            
            st.markdown("#### Step 1: Original Polynomial")
            poly_str = " + ".join([f"{coeffs_z[i]:.3f}z^{i}" if i > 0 else f"{coeffs_z[i]:.3f}" 
                                  for i in range(len(coeffs_z))])
            st.latex(f"Q(z) = {poly_str}")
            
            st.markdown("#### Step 2: Substitute z = (w+1)/(w-1)")
            
            try:
                w_coeffs, poly_w = bilinear_transform(coeffs_z)
                
                st.success("✅ Transformation successful!")
                st.latex(f"Q(w) = {sp.latex(poly_w)}")
                
                st.markdown("#### Step 3: Polynomial in w (standard form)")
                w_poly_str = " + ".join([f"{w_coeffs[i]:.4f}w^{i}" if i > 0 else f"{w_coeffs[i]:.4f}" 
                                        for i in range(len(w_coeffs))])
                st.latex(f"Q(w) = {w_poly_str}")
                
                # Visualize mapping
                st.markdown("#### Step 4: Verify Mapping Property")
                
                # Compute poles in both domains
                roots_z = np.roots(coeffs_z[::-1])
                roots_w = np.roots(w_coeffs[::-1])
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Z-plane
                draw_unit_circle(ax1)
                for root in roots_z:
                    mag = abs(root)
                    color = 'green' if mag < 1 else 'red' if mag > 1 else 'orange'
                    ax1.plot(root.real, root.imag, 'o', color=color, markersize=12,
                            markeredgewidth=2)
                ax1.set_title('Z-Plane (Original)', fontsize=13, fontweight='bold')
                
                # W-plane
                ax2.axhline(y=0, color='b', linewidth=2.5, label='Imaginary Axis')
                ax2.axvline(x=0, color='k', linewidth=2.5, label='Stability Boundary')
                ax2.fill_betweenx([-2, 2], -2, 0, color='green', alpha=0.2, label='Stable (LHP)')
                ax2.fill_betweenx([-2, 2], 0, 2, color='red', alpha=0.2, label='Unstable (RHP)')
                
                for root in roots_w:
                    color = 'green' if root.real < 0 else 'red' if root.real > 0 else 'orange'
                    ax2.plot(root.real, root.imag, 'o', color=color, markersize=12,
                            markeredgewidth=2)
                    ax2.text(root.real, root.imag + 0.1, f'({root.real:.2f}, {root.imag:.2f})',
                            ha='center', fontsize=8)
                
                ax2.set_xlabel('Real Part (σ)', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Imaginary Part (jω)', fontsize=12, fontweight='bold')
                ax2.set_title('W-Plane (Transformed)', fontsize=13, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.legend(fontsize=9)
                ax2.set_xlim(-2, 2)
                ax2.set_ylim(-2, 2)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Verify mapping
                st.markdown("#### Verification Table")
                verification_data = []
                for i, (z_root, w_root) in enumerate(zip(roots_z, roots_w)):
                    z_status = "Inside (Stable)" if abs(z_root) < 1 else \
                              "Outside (Unstable)" if abs(z_root) > 1 else "On Circle (Marginal)"
                    w_status = "LHP (Stable)" if w_root.real < 0 else \
                              "RHP (Unstable)" if w_root.real > 0 else "Imag Axis (Marginal)"
                    
                    verification_data.append({
                        "Root": f"Root {i+1}",
                        "z-value": f"{z_root:.4f}",
                        "|z|": f"{abs(z_root):.4f}",
                        "z-Status": z_status,
                        "w-value": f"{w_root:.4f}",
                        "Re(w)": f"{w_root.real:.4f}",
                        "w-Status": w_status,
                        "Match": "✅" if (z_status.startswith("Inside") == w_status.startswith("LHP")) else "❌"
                    })
                
                df = pd.DataFrame(verification_data)
                st.dataframe(df, use_container_width=True)
                
                if all(d["Match"] == "✅" for d in verification_data):
                    st.success("✅ Mapping verified correctly! Inside unit circle ↔ Left-half plane")
                
            except Exception as e:
                st.error(f"Error in transformation: {str(e)}")
        
        st.info("""
        **Sources:**  
        - Chakrabortty Ch.7, Section 7.3, pp. 234-236  
        - DCS.pdf, Bilinear Transformation section  
        - courssystèmes..., mapping proof
        """)

# ═══════════════════════════════════════════════════════════════════
# DEMO 5: ROUTH CRITERION IN W-PLANE
# ═══════════════════════════════════════════════════════════════════

elif "Demo 5" in demo_selection:
    st.markdown("## 🔧 Demo 5: Routh-Hurwitz Criterion in w-plane")
    st.markdown("Apply Routh criterion after bilinear transformation")
    
    st.markdown("""
    <div class="info-box">
        <h4>📚 Routh Criterion (Source: Chakrabortty Ch.7, pp. 236-239)</h4>
        <p><strong>Procedure:</strong></p>
        <ol>
            <li>Transform Q(z) → Q(w) using bilinear transform</li>
            <li>Construct Routh array for Q(w)</li>
            <li>Check first column for sign changes</li>
            <li>Zero sign changes → Stable; n sign changes → n unstable poles</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ System Input")
        
        method = st.radio("Input method:", ["Direct w-polynomial", "From z-polynomial"])
        
        if method == "From z-polynomial":
            st.markdown("**Characteristic equation in z:**")
            
            order = st.selectbox("Order:", [2, 3], index=0, key="routh_order")
            
            coeffs_z = []
            for i in range(order + 1):
                val = st.number_input(f"a_{i} (z^{i} coefficient):", 
                                     value=1.0 if i == order else 0.2 if i == 1 else -0.5,
                                     format="%.4f", key=f"routh_z_{i}")
                coeffs_z.append(val)
            
            if st.button("Transform to w"):
                w_coeffs, poly_w = bilinear_transform(coeffs_z)
                st.session_state.w_coeffs = w_coeffs
                st.success("✅ Transformed! Scroll down to see Routh array.")
        
        else:
            st.markdown("**Enter Q(w) coefficients directly:**")
            
            order = st.selectbox("Order:", [2, 3, 4], index=1, key="routh_w_order")
            
            w_coeffs = []
            for i in range(order + 1):
                val = st.number_input(f"a_{i} (w^{i} coefficient):", 
                                     value=1.0 if i == order else 0.0,
                                     format="%.4f", key=f"routh_w_direct_{i}")
                w_coeffs.append(val)
            
            st.session_state.w_coeffs = w_coeffs
    
    with col2:
        st.markdown("### 📊 Routh Array Analysis")
        
        if 'w_coeffs' in st.session_state:
            w_coeffs = st.session_state.w_coeffs
            
            st.markdown("#### Polynomial in w:")
            w_poly_str = " + ".join([f"{w_coeffs[i]:.4f}w^{i}" for i in range(len(w_coeffs))])
            st.latex(f"Q(w) = {w_poly_str}")
            
            if st.button("🚀 Generate Routh Array", type="primary"):
                
                try:
                    routh, stable, first_col = routh_array(w_coeffs)
                    
                    st.markdown("#### Routh Array:")
                    
                    # Display array
                    rows_labels = [f"w^{len(w_coeffs)-1-i}" for i in range(routh.shape[0])]
                    routh_df = pd.DataFrame(routh, index=rows_labels)
                    routh_df = routh_df.applymap(lambda x: f"{x:.6f}" if x != 0 else "--")
                    st.dataframe(routh_df, use_container_width=True)
                    
                    # Highlight first column
                    st.markdown("#### First Column Analysis:")
                    first_col_data = {
                        "Row": rows_labels[:len(first_col)],
                        "Value": [f"{x:.6f}" for x in first_col],
                        "Sign": ["+" if x > 0 else "-" if x < 0 else "0" for x in first_col]
                    }
                    df_first = pd.DataFrame(first_col_data)
                    st.table(df_first)
                    
                    # Count sign changes
                    sign_changes = 0
                    for i in range(len(first_col) - 1):
                        if first_col[i] * first_col[i+1] < 0:
                            sign_changes += 1
                    
                    st.markdown(f"**Number of sign changes:** {sign_changes}")
                    
                    # Verdict
                    if stable:
                        st.markdown("""
                        <div class="success-box">
                            <h3>✅ SYSTEM IS STABLE</h3>
                            <p>All elements in first column have the same sign.</p>
                            <p>Zero sign changes → All poles in left-half w-plane</p>
                            <p>→ All poles inside unit circle in z-plane</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="danger-box">
                            <h3>❌ SYSTEM IS UNSTABLE</h3>
                            <p>{sign_changes} sign change(s) in first column.</p>
                            <p>→ {sign_changes} pole(s) in right-half w-plane</p>
                            <p>→ {sign_changes} pole(s) outside unit circle in z-plane</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Verify by computing roots
                    st.markdown("#### Verification:")
                    roots_w = np.roots(w_coeffs[::-1])
                    
                    fig, ax = plt.subplots(figsize=(10, 7))
                    
                    ax.axhline(y=0, color='k', linewidth=1)
                    ax.axvline(x=0, color='b', linewidth=2.5, label='Stability Boundary')
                    ax.fill_betweenx([ax.get_ylim()[0], ax.get_ylim()[1]], 
                                    ax.get_xlim()[0], 0, color='green', alpha=0.2, label='Stable (LHP)')
                    ax.fill_betweenx([ax.get_ylim()[0], ax.get_ylim()[1]], 
                                    0, ax.get_xlim()[1], color='red', alpha=0.2, label='Unstable (RHP)')
                    
                    for i, root in enumerate(roots_w):
                        color = 'green' if root.real < -0.01 else 'red' if root.real > 0.01 else 'orange'
                        ax.plot(root.real, root.imag, 'o', color=color, markersize=14,
                               markeredgewidth=2.5, label=f'w_{i+1} = {root:.4f}')
                    
                    ax.set_xlabel('Real Part (σ)', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Imaginary Part (jω)', fontsize=12, fontweight='bold')
                    ax.set_title('Pole Locations in W-Plane', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='best', fontsize=10)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Summary table
                    summary_data = []
                    for i, root in enumerate(roots_w):
                        status = "LHP (Stable)" if root.real < -0.01 else \
                                "RHP (Unstable)" if root.real > 0.01 else \
                                "Imag Axis (Marginal)"
                        summary_data.append({
                            "Pole": f"w_{i+1}",
                            "Value": f"{root:.4f}",
                            "Real Part": f"{root.real:.4f}",
                            "Status": status
                        })
                    
                    df_summary = pd.DataFrame(summary_data)
                    st.table(df_summary)
                    
                except Exception as e:
                    st.error(f"Error in Routh analysis: {str(e)}")
        
        else:
            st.info("👈 Enter polynomial coefficients and transform to begin analysis")
        
        st.info("""
        **Sources:**  
        - Chakrabortty Ch.7, Examples 7.3, pp. 237-239  
        - DCS.pdf, Routh stability examples  
        - Standard Routh-Hurwitz criterion (prerequisite)
        """)

# ═══════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px;'>
        <p><strong>📚 Digital Control Systems - Week 6: Stability Analysis I</strong></p>
        <p>Dr. Ali Al-Ghanimi | Electrical Engineering | University of Kufa</p>
        <p>📧 ali.alghanimi@uokufa.edu.iq | 🏢 EE Building, Room 301</p>
        <hr style="width: 50%; margin: 1rem auto;">
        <p style="font-size: 0.9rem;"><strong>📖 Key Sources:</strong></p>
        <p style="font-size: 0.85rem;">
            Chakrabortty et al., Digital Control System Analysis & Design, Ch.7<br>
            DigitalControlTextBook.pdf, Chapter 4<br>
            DCS.pdf (Jury Test & Bilinear Transformation)
        </p>
        <p style="font-size: 0.9rem; color: #666; margin-top: 1rem;'>
            © 2025 University of Kufa | All rights reserved
        </p>
    </div>
""", unsafe_allow_html=True)
