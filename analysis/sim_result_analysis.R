library(tidyverse)
#devtools::install_github('flinder/flindR')
library(flindR)

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
    scale_color_manual(values = pe$colors, name = '') +
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
