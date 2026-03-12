library(ggplot2)

result_path = r"(../data/results/)"
file_names = list.files(result_path, pattern = "\\.csv$", full.names = TRUE)

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
    "log_sigma" = 0
)

long = long |> 
    dplyr::mutate(
        bias = estimate - reference_values[parameter], 
        rel_bias = bias / estimate
    )


# Result Plots ------------------------------------------------------------
selected_error = "ePIT"
selected_error_index = 2
parameter_names = c("beta0", "beta1", "beta2", "beta3")
plot_dir = "../images/2026_03_10 Zwischenergebnisse/"
plot_width = 8
plot_height = 5

error_variances = df |> dplyr::filter(error == !!selected_error) |> dplyr::pull(error_variance) |> unique()
selected_error_variance = error_variances[selected_error_index]
print(selected_error_variance)

plotfile_name = paste0("bias_",  selected_error, "_", selected_error_variance, ".png")
# Parameter Estimate Plots
long |> 
    dplyr::filter(error == !!selected_error, error_variance == !!selected_error_variance) |> 
    dplyr::filter(parameter %in% !!parameter_names) |> 
    ggplot() + 
    geom_boxplot(aes(y = estimate, x = model, color = model)) + 
    facet_wrap(~parameter, scales = "free")
# Bias Plot
long |> 
    dplyr::filter(error == !!selected_error, error_variance == !!selected_error_variance) |> 
    dplyr::filter(parameter %in% !!parameter_names) |> 
    ggplot() + 
    geom_boxplot(aes(y = bias, x = model, color = model)) + 
    geom_hline(yintercept = 0) + 
    facet_wrap(~parameter, scales = "free")
ggsave(paste0(plot_dir, plotfile_name), width = plot_width, height = plot_height)
long |> 
    dplyr::filter(error == !!selected_error, error_variance == !!selected_error_variance) |> 
    dplyr::filter(parameter %in% !!parameter_names) |> 
    ggplot() + 
    geom_boxplot(aes(y = rel_bias, x = model, color = model)) + 
    geom_hline(yintercept = 0) + 
    ylim(-1, 1) + 
    facet_wrap(~parameter, scales = "free")
ggsave(paste0(plot_dir, "rel_", plotfile_name), width = plot_width, height = plot_height)
# Rhat Plot
long |> 
    dplyr::filter(error == !!selected_error, error_variance == !!selected_error_variance) |> 
    dplyr::filter(!parameter %in% !!parameter_names, !parameter %in%  c("log_sigma", "rhat_log_sigma")) |> 
    ggplot() + 
    geom_boxplot(aes(y = estimate, x = model, color = model)) + 
    geom_hline(yintercept = 1) + 
    facet_wrap(~parameter, scales = "free")
ggsave(paste0(plot_dir, "rhat_", plotfile_name), width = plot_width, height = plot_height)
# Log Sigma
long |> 
    dplyr::filter(error == !!selected_error, error_variance == !!selected_error_variance) |> 
    dplyr::filter(parameter %in% c("log_sigma", "rhat_log_sigma")) |> 
    ggplot() + 
    geom_boxplot(aes(y = estimate, x = model, color = model)) + 
    facet_wrap(~parameter, scales = "free", )

    