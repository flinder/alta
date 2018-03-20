library(tidyverse)
#devtools::install_github('flinder/flindR')
library(flindR)
library(reshape2)

# ==============================================================================
# CONFIG
# ==============================================================================
pe = plot_elements()
DATA_DIR = "../data/runs/tweets/"
BATCH_SIZE = 20 # Size of AL batches should probably be loaded form config.yaml

plot_theme = pe$theme +
    theme(strip.text.x = element_text(size = 14))

# ==============================================================================
# Data prep
# ==============================================================================
files = list.files(DATA_DIR, recursive = TRUE, pattern = '\\d.csv')
proc_file = function(filename) {
    run = as.integer(unlist(strsplit(filename, '/'))[1])
    comp = unlist(strsplit(filename, '_'))
    algo = unlist(strsplit(comp[1], '/'))[2]
    data = suppressMessages(read_csv(paste0(DATA_DIR, filename))) %>%
        mutate(algo = algo,
               balance = as.numeric(gsub('\\.csv', '', comp[4])),
               run = run)
}
data = do.call(rbind, lapply(files, proc_file)) %>%
    mutate(f1_score = ifelse(is.na(f1), 0, f1),
           precision = ifelse(is.na(p), 0, p),
           recall = ifelse(is.na(r), 0, r),
           balance = paste0("Balance: ", balance))

# ==============================================================================
# Visualize results
# ==============================================================================
# F1 score
ggplot(data, aes(x = batch * BATCH_SIZE, y = f1, color = algo,
                 linetype = algo)) +
    facet_wrap(~balance, scales = 'free') +
    geom_point(size = 0.1, alpha = 0.1) + 
    geom_smooth() +
    scale_color_manual(values = pe$colors[-1], name = 'Labeling\nAlgorithm') +
    scale_linetype(name = 'Labeling\nAlgorithm') +
    ylab('F1 Score') + xlab('# labeled samples') +
    plot_theme
ggsave('../paper/figures/tweets_f1.png', width = pe$p_width, 
       height = 0.7*pe$p_width)
ggsave('../presentation/figures/tweets_f1.png', width = pe$p_width, 
       height = 0.7*pe$p_width)

# Precision
ggplot(data, aes(x = batch * BATCH_SIZE, y = precision, color = algo,
                 linetype = algo)) +
    facet_wrap(~balance, scales = 'free') +
    geom_point(size = 0.1, alpha = 0.1) + 
    geom_smooth() +
    scale_color_manual(values = pe$colors[-1], name = 'Labeling\nAlgorithm') +
    scale_linetype(name = 'Labeling\nAlgorithm') +
    ylab('Precision') + xlab('# labeled samples') +
    plot_theme
ggsave('../paper/figures/tweets_precision.png', width = pe$p_width, 
       height = 0.7*pe$p_width)
ggsave('../presentation/figures/tweets_precision.png', width = pe$p_width, 
       height = 0.7*pe$p_width)

# Recall
ggplot(data, aes(x = batch * BATCH_SIZE, y = recall, color = algo,
                 linetype = algo)) +
    facet_wrap(~balance, scales = 'free') +
    geom_point(size = 0.1, alpha = 0.1) + 
    geom_smooth() +
    scale_color_manual(values = pe$colors[-1], name = 'Labeling\nAlgorithm') +
    scale_linetype(name = 'Labeling\nAlgorithm') +
    ylab('Recall') + xlab('# labeled samples') +
    plot_theme
ggsave('../paper/figures/tweets_recall.png', width = pe$p_width, 
       height = 0.7*pe$p_width)
ggsave('../presentation/figures/tweets_recall.png', width = pe$p_width, 
       height = 0.7*pe$p_width)

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


