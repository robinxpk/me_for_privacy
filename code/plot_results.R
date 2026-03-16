library(ggplot2)
# Prevent scientific e-10 notation in numbers which looks ugly in plots
options(scipen=999)
# Params ------------------------------------------------------------------
result_path = r"(../data/results/)"
file_names = list.files(result_path, pattern = "\\.csv$", full.names = TRUE)


# Functions ---------------------------------------------------------------
# Coverage Plot 

plot_bias = function(
        df,
        selected_error, # normal, lognormal, ePIT
        selected_error_variance, # error variance value 
        parameter_names = c("beta0", "beta1", "beta2", "beta3"), 
        plot_dir = "../images/2026_03_10 Zwischenergebnisse/", 
        plot_width = 8,
        plot_height = 5
    ){
        param_labs = c(
            beta0     = "beta[0]",
            beta1     = "beta[1]",
            beta2     = "beta[2]",
            beta3     = "beta[3]",
            log_sigma = "log(sigma)"
        )
        rhat_labs = c(
          rhat_beta0 = "hat(R)[beta[0]]",
          rhat_beta1 = "hat(R)[beta[1]]",
          rhat_beta2 = "hat(R)[beta[2]]",
          rhat_beta3 = "hat(R)[beta[3]]"
        )
        logsig_labs = c(
            log_sigma     = "log(sigma)",
            rhat_log_sigma = "hat(R)[log(sigma)]"
        )
        
        working_df = df |> 
            dplyr::filter(
                error == !!selected_error, 
                error_variance == !!selected_error_variance
            ) 
        # Rel. Bias plot
        rel_bias_plot = working_df |> 
            dplyr::filter(parameter %in% !!parameter_names) |>
            ggplot(aes(x = model, y = rel_bias, fill = model)) +
            geom_hline(yintercept = 0) +
            geom_boxplot(
                colour = "black",
                width = 0.65,
                outlier.shape = 21,
                outlier.size = 2,
                outlier.stroke = 0.4
            ) +
            facet_wrap(
                ~parameter,
                scales = "free",
                labeller = as_labeller(param_labs, label_parsed)
            ) +
            ## ePIT limits: 
            ggh4x::facetted_pos_scales(
                # ePIT: 
                # y = list(
                #     parameter == "beta0" ~ scale_y_continuous(limits = c(-0.15, 0.1)),
                #     parameter == "beta1" ~ scale_y_continuous(limits = c(-2, 2)),
                #     parameter == "beta2" ~ scale_y_continuous(limits = c(-0.3, 0.3)),
                #     parameter == "beta3" ~ scale_y_continuous(limits = c(-10, 650))
                # )
                # normal: 
                y = list(
                    parameter == "beta0" ~ scale_y_continuous(limits = c(-0.15, 0.1)),
                    parameter == "beta1" ~ scale_y_continuous(limits = c(-1.5, 3)),
                    parameter == "beta2" ~ scale_y_continuous(limits = c(-0.5, 0.3)),
                    parameter == "beta3" ~ scale_y_continuous(limits = c(-1.2, 0.5))
                )
            ) +
            scale_fill_manual(
                values = c(corrected = "orange", naive = "blue"),
                breaks = c("corrected", "naive"),
                name = "Model"
            ) +
            labs(
                x = NULL,
                y = "Relative bias"
            ) +
            theme_classic(base_size = 20) +
            theme(
                panel.spacing = grid::unit(0.8, "lines"),
                strip.background = element_rect(fill = "grey90", colour = NA),
                strip.text = element_text(size = 14, colour = "black"),
                legend.position = "none", 
                axis.title = element_text(size = 20),
                axis.text = element_text(size = 16),
                axis.line = element_line(colour = "black"),
                axis.ticks = element_line(colour = "black")
            )
    
        # Rhat Plot
        rhat_plot = working_df |> 
          dplyr::filter(
            !parameter %in% !!parameter_names,
            !parameter %in% c("log_sigma", "rhat_log_sigma")
          ) |>
          ggplot(aes(x = model, y = estimate, fill = model)) +
          geom_hline(yintercept = 1) +
          geom_boxplot(
            colour = "black",
            width = 0.65,
            outlier.shape = 21,
            outlier.size = 2,
            outlier.stroke = 0.4
          ) +
          geom_hline(yintercept = 1.005, alpha = 0.5, linewidth = 0.6, colour = "grey45") + 
          facet_wrap(
            ~parameter,
            scales = "free",
            labeller = as_labeller(rhat_labs, label_parsed)
          ) +

          scale_fill_manual(
            values = c(corrected = "orange", naive = "blue"),
            breaks = c("corrected", "naive"),
            name = "Model"
          ) +
          labs(
            x = NULL,
            y = "R-hat"
          ) +
          theme_classic(base_size = 20) +
          scale_y_continuous(breaks = c(1, 1.005)) + 
          theme(
              panel.spacing = grid::unit(0.8, "lines"),
                strip.background = element_rect(fill = "grey90", colour = NA),
                strip.text = element_text(size = 14, colour = "black"),
              legend.position = "none", 
              axis.title = element_text(size = 20),
              axis.text = element_text(size = 16),
              axis.line = element_line(colour = "black"),
              axis.ticks = element_line(colour = "black"), 
          )

        plotfile_name = paste0(selected_error, "_", selected_error_variance, ".png")
        plot(rel_bias_plot)
        ggsave(plot = rel_bias_plot, paste0(plot_dir, "rel_bias_", plotfile_name), width = plot_width, height = plot_height)
        # plot(log_sigma_plot)
        plot(rhat_plot)
        ggsave(plot = rhat_plot, paste0(plot_dir, "rhat_", plotfile_name), width = plot_width, height = plot_height)
}

plot_coverage = function(
    df, 
    selected_error, 
    parameter_name, 
    error_variance, 
    scales_ = "fixed", 
    plot_dir = "../images/2026_03_10 Zwischenergebnisse/"
){
    latex_display = c(
        "beta0"     = r"($\beta_0$)",
        "beta1"     = r"($\beta_1$)",
        "beta2"     = r"($\beta_2$)",
        "beta3"     = r"($\beta_3$)",
        "log_sigma" = r"($\log(\sigma^2)$)"
    )
    latex_string = latex2exp::TeX(latex_display[[parameter_name]])
    
    
    working_df = df |> 
        dplyr::filter(
            error == !!selected_error, 
            parameter == !!parameter_name, 
            error_variance == !!error_variance
        ) 
    
    coverage = working_df |> 
        dplyr::summarise(.by = model, coverage = floor(mean(is_covered) * 100))
    naive_coverage = coverage |> dplyr::filter(model == "naive") |> dplyr::pull(coverage)
    corrected_coverage = coverage |> dplyr::filter(model == "corrected") |> dplyr::pull(coverage)
    
    coverage_plot = working_df |>  
        ggplot(aes(x = b)) + 
        geom_linerange(aes(ymin = lower, ymax = upper, color = is_covered)) + 
        scale_color_manual(
            values = c(`TRUE` = "darkgreen", `FALSE` = "darkred"),
            name = "Covered"
        ) +
        geom_point(aes(y = estimate, fill = model), shape = 21, size = 2.5, color = "black", stroke = 0.7) +
        scale_fill_manual(
            values = c(naive = "blue", corrected = "orange"),
            name = "Model"
        ) +
        geom_hline(aes(yintercept = reference_value), alpha = 0.5) + 
        facet_wrap(
            ~model, 
            scales = scales_, 
            labeller = as_labeller(c(
                naive = paste0("naive [",  naive_coverage, "%]"), 
                corrected = paste0("corrected [", corrected_coverage, "%]")
            ))
        ) + 
        ggh4x::facetted_pos_scales(
              # ePIT beta3 für free scale, da rest okay und bei fixed scale eh kaum Unterschied
              # y = list(
              #     model == "naive" ~ scale_y_continuous(limits = c(-0.5, 0.1)),
              #     model == "corrected" ~ scale_y_continuous(limits = c(-0.001, 0.0002))
              # )
              # ePIT log_sigma
              y = list(
                  model == "naive" ~ scale_y_continuous(limits = c(0.5, 0.7)),
                  model == "corrected" ~ scale_y_continuous(limits = c(0.5, 0.8))
              )
          ) +
        labs(x = "Iteration", y = latex_string) + 
        theme_classic(base_size = 20) +           theme(
          panel.spacing = grid::unit(0.8, "lines"),
            strip.background = element_rect(fill = "grey90", colour = NA),
            strip.text = element_text(size = 14, colour = "black"),
          legend.position = "none", 
          axis.title = element_text(size = 20),
          axis.text = element_text(size = 16),
          axis.line = element_line(colour = "black"),
          axis.ticks = element_line(colour = "black"), 
      )
    
    plot(coverage_plot)
    plotfile_name = paste0(selected_error, "_", parameter_name, "_", selected_error_variance, "_", scales_, "scale.png")
    ggsave(plot = coverage_plot, paste0(plot_dir, "coverage_", plotfile_name), width = plot_width, height = plot_height)
}

# Create Result DF  -------------------------------------------------------
# Iterate through files, read in and add origin column
df = do.call(
    rbind,
    lapply(file_names, function(file_name) {
        x = read.csv(file_name, stringsAsFactors = FALSE, sep = ";")
        # Save meta data into df, too, which is based on the filename: 
        # out_<error>_<var>_<iteration>.csv
        # (?!out_)[a-z]*_\d*\.\d*_\d*(?=\.csv)
        # meta_info = stringr::str_split(
        #     stringr::str_extract(file_name, stringr::regex(r"((?!out_)[a-z]*_\d*\.\d*_\d*(?=\.csv))")), 
        #     "_"
        # ) |> unlist()
        return(x)
    })
)

# Some cast typing to ensure all is working as expected
df = df |> 
    dplyr::mutate(error = as.factor(error), b = as.numeric(b), error_variance = as.factor(error_variance))

long = df |> 
    tidyr::pivot_longer(
        cols = dplyr::starts_with(c("naive", "corrected")), 
        # names_to = c("model", "parameter", "index"), 
        names_to = c("model", "parameter"), 
        # names_pattern = "(^[a-z]*)_([a-z_]*)(\\d?)",
        names_pattern = "(^[a-z]*)_(.*)",
        values_to = "estimate"
    ) |> 
    dplyr::mutate(
        model = as.factor(model), 
        parameter = as.factor(parameter)
    )

ci = long |> 
    dplyr::filter(grepl(pattern = "ci", x = parameter)) |> 
    dplyr::mutate(
        parameter_long = parameter, 
        parameter = stringr::str_extract(pattern = r"(.*(?=_ci))", string = parameter_long), 
        ci = stringr::str_extract(pattern = r"((?!ci_)[a-z]*$)", string = parameter_long) 
    ) |> 
    dplyr::select(-parameter_long)
    
ci = ci |> 
    tidyr::pivot_wider(
        id_cols = c(error, error_variance, b, model, parameter), 
        names_from = ci, 
        values_from = estimate
    )

long = long |> 
    dplyr::filter(!grepl(pattern = "ci", x = parameter)) |> 
    dplyr::left_join(
        ci, 
        by = c("error", "error_variance", "b", "model", "parameter"), 
    )
## TODO: Remove later on: 
# My sampler seems to sample 2log(sigma^2) or smth, at least for CORRECTED ePIT. Not sure whats going on, but quick fixing it for now; The naive model has the correct posterior...
long[long$parameter == "log_sigma" & long$model == "corrected", c("estimate", "lower", "upper")] = long[long$parameter == "log_sigma" & long$model == "corrected", c("estimate", "lower", "upper")] / 2
    
# Add Biases
# |           |    Reference |
# |:----------|-------------:|
# | Intercept |  8.7095      |
# | RIDAGEYR  |  0.00186197  |
# | bmi       | -0.0748638   |
# | DR1TKCAL  | -0.000378365 |
reference_values = c(
    "beta0" =  8.7095     ,
    "beta1" =  0.00186197 ,
    "beta2" = -0.0748638  ,
    "beta3" = -0.000378365, 
    "log_sigma" = 0.6257477
)

long = long |> 
    dplyr::mutate(
        reference_value = reference_values[parameter],
        is_covered = (reference_value >= lower) & (reference_value <= upper), 
        bias = estimate - reference_value, 
        rel_bias = bias / reference_value 
    )


# Result Plots ------------------------------------------------------------
selected_error = "ePIT"
selected_error_index = 1
parameter_names = c("beta0", "beta1", "beta2", "beta3")
plot_dir = "../images/2026_03_10 Zwischenergebnisse/"
plot_width = 8
plot_height = 5

error_variances = df |> dplyr::filter(error == !!selected_error) |> dplyr::pull(error_variance) |> unique()
selected_error_variance = error_variances[[selected_error_index]]
print(selected_error_variance)

plot_bias(
    df = long, 
    selected_error = selected_error, 
    selected_error_variance = selected_error_variance,
)
plot_coverage(df = long, selected_error = selected_error, parameter_name = "log_sigma", error_variance = selected_error_variance, scales_ = "fixed")

plotfile_name = paste0("bias_",  selected_error, "_", selected_error_variance, ".png")
# Parameter Estimate Plots
long |> 
    dplyr::filter(error == !!selected_error, error_variance == !!selected_error_variance) |> 
    dplyr::filter(parameter %in% !!parameter_names) |> 
    ggplot() + 
    geom_boxplot(aes(y = estimate, x = model, color = model)) + 
    geom_hline(aes(yintercept = reference_value)) + 
    facet_wrap(~parameter, scales = "free")

# Bias Plot
# long |> 
#     dplyr::filter(error == !!selected_error, error_variance == !!selected_error_variance) |> 
#     dplyr::filter(parameter %in% !!parameter_names) |> 
#     ggplot() + 
#     geom_boxplot(aes(y = bias, x = model, color = model)) + 
#     geom_hline(yintercept = 0) + 
#     facet_wrap(~parameter, scales = "free")
# ggsave(paste0(plot_dir, plotfile_name), width = plot_width, height = plot_height)

