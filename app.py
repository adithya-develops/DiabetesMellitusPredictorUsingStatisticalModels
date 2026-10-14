        min_size = st.slider("Minimum observations per group", 2, 100, 20)

    try:
        fstat, pval, dfb, dfw, eta, summary = one_way_anova(df, nums[numerical_label], groups[group_label], min_size)
        significant = pval < 0.05
        conclusion = (
            f"There is statistically significant evidence that mean {numerical_label.lower()} differs across {group_label.lower()} groups."
            if significant else
            f"There is not enough evidence at α = 0.05 to say that mean {numerical_label.lower()} differs across {group_label.lower()} groups."
        )

        st.markdown("<div style='height:17px'></div>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4, gap="medium")
        # η² is often very small in large datasets. Fixed three-decimal formatting
        # made legitimate non-zero effects appear as 0.000, so adapt precision.
        if not np.isfinite(eta):
            eta_display = "—"
            eta_description = "Not available for this test"
        elif eta >= 0.01:
            eta_display = f"{eta:.3f}"
            eta_description = f"{eta * 100:.3g}% of variance explained"
        elif eta >= 0.0001:
            eta_display = f"{eta:.4f}"
            eta_description = f"{eta * 100:.3g}% of variance explained"
        elif eta > 0:
            eta_display = f"{eta:.2e}"
            eta_description = f"{eta * 100:.2g}% of variance explained"
        else:
            eta_display = "0.0000"
            eta_description = "0% of variance explained"

        with m1: card_metric("F statistic", f"{fstat:.3f}", f"df {dfb} / {dfw}", "lime")
        with m2: card_metric("p-value", f"{pval:.3e}", "α = 0.05", "mint")
        with m3: card_metric("η² effect size", eta_display, eta_description, "blue")
        with m4: card_metric("Groups", str(len(summary)), "Included in test", "coral")

        with st.container(border=True):
            section_heading("Interpretation", "H₀: all group means are equal. H₁: at least one mean differs.", "Test result", "α = 0.05")
            st.markdown(f'<div class="conclusion">{html.escape(conclusion)}</div>', unsafe_allow_html=True)

        l, r = st.columns([.95, 1.2], gap="large")
        with l:
            with st.container(border=True):
                section_heading("Group summary", "Sample size, mean and standard deviation.", "Output")
                st.dataframe(summary.round({"Mean": 3, "SD": 3}), use_container_width=True, hide_index=True)
        with r:
            with st.container(border=True):
                section_heading(f"{numerical_label} by {group_label.lower()}", "Distribution across selected groups.", "Distribution")
                plot = df[[nums[numerical_label], groups[group_label]]].dropna()
                if len(plot) > 12000:
                    plot = plot.sample(12000, random_state=42)
                fig = px.box(plot, x=groups[group_label], y=nums[numerical_label], points=False,
                             color_discrete_sequence=["#d8f56c"], labels={nums[numerical_label]: numerical_label, groups[group_label]: group_label})
                fig.update_traces(line_color="#d8f56c", fillcolor="rgba(216,245,108,.13)")
                st.plotly_chart(plotly_theme(fig, 400), use_container_width=True, config={"displayModeBar": False})
    except Exception as exc:
        st.error(f"ANOVA could not be calculated: {exc}")

# -----------------------------------------------------------------------------
# Dataset visualisations
# -----------------------------------------------------------------------------
else:
    page_header("Visual analysis", "Distributions and relationships in the dataset.", page_indexes[page], "Visuals")
    visual1, visual2 = st.columns(2, gap="large")
    with visual1:
        with st.container(border=True):
            section_heading("Blood glucose distribution", "Distribution split by diabetes outcome.", "01 / Distribution")
            fig = px.histogram(df, x="blood_glucose_level", color="diabetes", nbins=45, marginal="box",
                               labels={"blood_glucose_level": "Blood glucose", "diabetes": "Diabetes"},
                               color_discrete_map={0: "#4b5962", 1: "#d8f56c"}, opacity=.88)
            fig.update_layout(legend_title_text="Diabetes")
            st.plotly_chart(plotly_theme(fig, 390), use_container_width=True, config={"displayModeBar": False})
    with visual2:
        with st.container(border=True):
            section_heading("BMI vs blood glucose", "Patient-level relationship, sampled for readability.", "02 / Relationship")
            scatter_sample = df.sample(min(5000, len(df)), random_state=7)
            fig = px.scatter(scatter_sample, x="bmi", y="blood_glucose_level", color="diabetes", opacity=.58,
                             labels={"bmi": "BMI", "blood_glucose_level": "Blood glucose", "diabetes": "Diabetes"},
                             color_discrete_map={0: "#6c7c85", 1: "#d8f56c"})
            fig.update_traces(marker=dict(size=5, line=dict(width=0)))
            st.plotly_chart(plotly_theme(fig, 390), use_container_width=True, config={"displayModeBar": False})

    visual3, visual4 = st.columns(2, gap="large")
    with visual3:
        with st.container(border=True):
            section_heading("Smoking history", "Observed diabetes percentage in each category.", "03 / Conditional rate")
            smoking_rates = df.groupby("smoking_history", as_index=False).diabetes.mean()
            smoking_rates["rate"] = smoking_rates.diabetes * 100
            smoking_rates = smoking_rates.sort_values("rate")
            fig = px.bar(smoking_rates, x="rate", y="smoking_history", orientation="h",
                         text=smoking_rates["rate"].map(lambda x: f"{x:.2f}%"),
                         labels={"rate": "Probability (%)", "smoking_history": "Smoking history"})
            fig.update_traces(marker_color="#73d8b4", textposition="outside", cliponaxis=False, marker_line_width=0)
            fig.update_layout(xaxis=dict(range=[0, max(smoking_rates.rate) * 1.2]))
            st.plotly_chart(plotly_theme(fig, 350), use_container_width=True, config={"displayModeBar": False})
    with visual4:
        with st.container(border=True):
            section_heading("Numerical correlation", "Pearson correlation between numeric variables.", "04 / Correlation")
            correlation = df[["age", "bmi", "HbA1c_level", "blood_glucose_level", "diabetes"]].corr()
            fig = px.imshow(correlation, text_auto=".2f", aspect="auto",
                            color_continuous_scale=["#182027", "#5b7a79", "#d8f56c"], zmin=-1, zmax=1)
            fig.update_traces(textfont=dict(color="#edf2f2", size=10))
            fig.update_layout(coloraxis_colorbar=dict(tickfont=dict(color="#a3afb4"), outlinewidth=0, thickness=8))
            st.plotly_chart(plotly_theme(fig, 350), use_container_width=True, config={"displayModeBar": False})

    with st.container(border=True):
        section_heading("Dataset sample", "Twenty rows for inspection. Statistical calculations use the full dataset.", "Raw records", "20 ROWS")
        st.dataframe(df.sample(20, random_state=42), use_container_width=True, hide_index=True)

add_footer()
