library(tidyverse)
#devtools::install_github('flinder/flindR')
library(flindR)
library(reshape2)

pe = plot_elements()
DATA_DIR = "../data/"
A_SIM_OUT = paste0(DATA_DIR, 'active_simulation_data.csv')
R_SIM_OUT = paste0(DATA_DIR, 'random_simulation_data.csv')


# ==============================================================================
# Data prep
# ==============================================================================
active = read_csv(A_SIM_OUT) %>% mutate(group = 'active')
random = read_csv(R_SIM_OUT) %>% mutate(group = 'random')

pdat = rbind(active, random)

# ==============================================================================
# Visualize results
# ==============================================================================
# Learning rate
ggplot(pdat, aes(x = batch, y = f1, color = group)) +
    facet_wrap(~label) +
    geom_line() +
   s cale_color_manual(values = pe$colors, name = '') +
    pe$theme    

# Support
ggplot(pdat, aes(x = batch, y = support, color = group)) +
    facet_wrap(~label) +
    geom_line() +
    scale_color_manual(values = pe$colors, name = '') +
    pe$theme    

# Learning rate by support
ggplot(pdat, aes(x = support, y = f1, color = group)) +
    facet_wrap(~label, scales = 'free') +
    geom_line() +
    scale_color_manual(values = pe$colors, name = '') +
    pe$theme    

# ==============================================================================
# Plots for presentation
# ==============================================================================

twitter_data = read_csv('../data/annotated_german_refugee_tweets.csv') %>%
    select(-text) %>%
    mutate(t_annotation = as.factor(ifelse(annotation == 1, 'relevant', 
                                         'irrelevant')))

# Demonstrate sparsity of relevant twitter data
ggplot(twitter_data) +
    geom_bar(aes(x = t_annotation, fill = t_annotation, color = t_annotation), 
             alpha = 0.5) + 
    scale_fill_manual(values = pe$colors, name = "", guide = F) +
    scale_color_manual(values = pe$colors, name = "", guide = F) +
    xlab('') + ylab('') +
    pe$theme
ggsave('../presentation/figures/twitter_proportion.png', width = pe$p_width,
       height = 0.7*pe$p_width)

# Demonstrate how many tweets have to be annotated to get X relevant labels
bs_iter = function(i) {
    p_pos = mean(sample(twitter_data$annotation, nrow(twitter_data), 
                        replace = T))
    return(seq(100, 1000, 10) / p_pos) 
}

bs_out = sapply(1:1000, bs_iter)

pdat = t(bs_out) %>%
    melt() %>% tbl_df()

ggplot(pdat) +
    geom_jitter(aes(x = Var2 * 10 + 100, y = value), size = 1, alpha = 0.01) +
    xlab("# of positively labeled samples") + 
    ylab("# of required labeled samples") +
    scale_y_continuous(labels = scales::comma) +
    pe$theme
ggsave('../presentation/figures/required_samples.png')
