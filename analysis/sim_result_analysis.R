library(tidyverse)
library(assertthat)
#devtools::install_github('flinder/flindR')
library(flindR)
library(reshape2)
library(yaml)
library(stringr)

# ==============================================================================
# CONFIG
# ==============================================================================
proj_conf <- yaml.load_file("../config.yaml")

BATCH_SIZE = 20 # Size of AL batches should probably be loaded form config.yaml
DATA_SETS = names(proj_conf$data_sets)
DATA_DIR = "../data/runs/"

pe = plot_elements()
plot_theme = pe$theme +
    theme(strip.text.x = element_text(size = 14))

# ==============================================================================
# Data prep
# ==============================================================================
proc_file = function(filename) {
    comp = unlist(strsplit(filename, '/'))
    run = as.integer(comp[5])
    comp_2 = unlist(strsplit(comp[6], '_'))
    algo = comp_2[1]
    balance = gsub('\\.csv', '', comp_2[4])
    data = suppressMessages(read_csv(filename)) %>%
        mutate(algo = algo, balance = balance, run = run)
}


# ==============================================================================
# Visualize results
# ==============================================================================

for(data_set in DATA_SETS) {
    
    cat('Processing ', data_set, '\n')    
    
    inpath = paste0(DATA_DIR, data_set) 
    files = list.files(inpath, recursive = TRUE, pattern = '\\d.csv$')
    files = paste0(inpath, '/', files)

    data = do.call(rbind, lapply(files, proc_file)) %>%
        mutate(f1_score = ifelse(is.na(f1), 0, f1),
               precision = ifelse(is.na(p), 0, p),
               recall = ifelse(is.na(r), 0, r),
               balance = paste0("Balance: ", balance))
     
   
    # F1 score
    ggplot(data, aes(x = batch * BATCH_SIZE, y = f1, color = algo,
                     linetype = algo)) +
        facet_wrap(~balance, scales = 'free') +
        geom_point(size = 0.1, alpha = 0.1) + 
        geom_smooth() +
        scale_color_manual(values = pe$colors, name = 'Labeling\nAlgorithm') +
        scale_linetype(name = 'Labeling\nAlgorithm') +
        ylab('F1 Score') + xlab('# labeled samples') +
        plot_theme
    ggsave(paste0('../paper/figures/',data_set ,'_f1.png'), width = pe$p_width, 
           height = 0.7*pe$p_width)
    ggsave(paste0('../presentation/figures/',data_set, '_f1.png'), 
           width = pe$p_width, height = 0.7*pe$p_width)
    
    # Precision
    ggplot(data, aes(x = batch * BATCH_SIZE, y = precision, color = algo,
                     linetype = algo)) +
        facet_wrap(~balance, scales = 'free') +
        geom_point(size = 0.1, alpha = 0.1) + 
        geom_smooth() +
        scale_color_manual(values = pe$colors, name = 'Labeling\nAlgorithm') +
        scale_linetype(name = 'Labeling\nAlgorithm') +
        ylab('Precision') + xlab('# labeled samples') +
        plot_theme
    ggsave(paste0('../paper/figures/', data_set, '_precision.png'), 
           width = pe$p_width, height = 0.7*pe$p_width)
    ggsave(paste0('../presentation/figures/', data_set, '_precision.png'), 
           width = pe$p_width, height = 0.7*pe$p_width)
    
    # Recall
    ggplot(data, aes(x = batch * BATCH_SIZE, y = recall, color = algo,
                     linetype = algo)) +
        facet_wrap(~balance, scales = 'free') +
        geom_point(size = 0.1, alpha = 0.1) + 
        geom_smooth() +
        scale_color_manual(values = pe$colors, name = 'Labeling\nAlgorithm') +
        scale_linetype(name = 'Labeling\nAlgorithm') +
        ylab('Recall') + xlab('# labeled samples') +
        plot_theme
    ggsave(paste0('../paper/figures/', data_set, '_recall.png'), 
           width = pe$p_width, height = 0.7*pe$p_width)
    ggsave(paste0('../presentation/figures/', data_set, '_recall.png'), 
           width = pe$p_width, height = 0.7*pe$p_width)
    
    # Visualize support growth
    d = filter(data, balance == 'Balance: 0.01') %>%
        group_by(batch, algo) %>%
        summarize(mean_f1 = mean(f1_score), mean_support = mean(support))
    #total_positives = proj_conf$data_sets$tweets$n_positive
    
    ggplot(d, aes(x = batch * BATCH_SIZE, y = mean_f1, color = algo,
                  size = mean_support)) + 
        geom_point(alpha = 0.4) + 
        #geom_line(size = 1) +
        scale_color_manual(values = pe$colors, name = 'Labeling\nAlgorithm') +
        scale_linetype(name = 'Labeling\nAlgorithm') +
        scale_size_continuous(name = 'Mean # of\npositives', range = c(0.5, 4)) +
        ylab('F1-Score (mean)') + xlab('# labeled samples') +
        plot_theme
    ggsave(paste0('../paper/figures/', data_set, 
                  '_f1_labeled_support_balance_0.01.png'), width = pe$p_width, 
           height = 0.7*pe$p_width)
    ggsave(paste0('../presentation/figures/', data_set, 
                  '_f1_labeled_support_balance_0.01.png'), 
       width = pe$p_width, height = 0.7*pe$p_width)
 
}

# ==============================================================================
# Pre-pocessing choices
# ==============================================================================

pre_proc_data = list()
i = 1 
for(data_set in DATA_SETS) {
     
    cat('Processing ', data_set, '\n')    
    
    inpath = paste0(DATA_DIR, data_set) 
    files = list.files(inpath, recursive = TRUE, pattern = '\\d.csv$')
    files = paste0(inpath, '/', files)

    pre_proc_data[[i]] = do.call(rbind, lapply(files, proc_file)) %>%
        select(batch, support, text__selector__fname, algo, run, f1) %>%
        mutate(balance = as.numeric(balance), data_set = data_set)
    i = i + 1
}
pp_dat = do.call(rbind, pre_proc_data) %>%
    filter(!is.na(text__selector__fname))

# Parse out all preprocessing options from text selector argument    
pp_info = pp_dat$text__selector__fname
pp_dat$token_type = gsub("'", '', str_extract(pp_info, "'.+'"))
pp_dat$gram_size = as.integer(gsub("'", '', str_extract(pp_info, "\\d")))
weight_stem_info = str_extract(pp_info, '(True|False)_(True|False)')
tfidf_str = sapply(weight_stem_info, function(x) unlist(strsplit(x, '_'))[1])  
pp_dat$tfidf = tfidf_str == 'True'
stem_str = sapply(weight_stem_info, function(x) unlist(strsplit(x, '_'))[2])  
pp_dat$stem = stem_str == 'True'
    
pp_dat = select(pp_dat, -text__selector__fname)  

## Tfidf
## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
plot_dat = group_by(pp_dat, data_set, tfidf) %>%
    summarize(count = n()) %>%
    group_by(data_set) %>%
    mutate(prop = count / sum(count))
ggplot(plot_dat) +
    geom_bar(aes(data_set, prop, fill = tfidf), stat = 'identity',
             position = 'dodge', alpha = 0.8, color = 'white') +
    scale_fill_manual(values = pe$colors, name = 'Token\nWeight', 
                      labels = c('Count', 'Tf-Idf')) +
    scale_x_discrete(labels = c('Twitter', 'Wikipedia')) +
    plot_theme + ylab('Optimal in Proportion of Runs') + xlab('')
ggsave('../paper/figures/preproc_tfidf.png', 
       width = pe$p_width, height = 0.7*pe$p_width)
ggsave('../presentation/figures/preproc_tfidf.png', 
       width = pe$p_width, height = 0.7*pe$p_width)

## Stemming Lemmatization
## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
plot_dat = group_by(pp_dat, data_set, stem) %>%
    summarize(count = n()) %>%
    group_by(data_set) %>%
    mutate(prop = count / sum(count))
ggplot(plot_dat) +
    geom_bar(aes(data_set, prop, fill = stem), stat = 'identity',
             position = 'dodge', alpha = 0.8, color = 'white') +
    scale_fill_manual(values = pe$colors, name = 'Normalize', 
                      labels = c('Lemmatize', 'Stem')) +
    scale_x_discrete(labels = c('Twitter', 'Wikipedia')) +
    plot_theme + ylab('Optimal in Proportion of Runs') + xlab('')
ggsave('../paper/figures/preproc_stem.png', 
       width = pe$p_width, height = 0.7*pe$p_width)
ggsave('../presentation/figures/preproc_stem.png', 
       width = pe$p_width, height = 0.7*pe$p_width)

## Token type
## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
plot_dat = group_by(pp_dat, data_set, token_type) %>%
    summarize(count = n()) %>%
    group_by(data_set) %>%
    mutate(prop = count / sum(count))
ggplot(plot_dat) +
    geom_bar(aes(data_set, prop, fill = token_type), stat = 'identity',
             position = 'dodge', alpha = 0.8, color = 'white') +
    scale_fill_manual(values = pe$colors, name = 'Token\nType',
                      labels = c('Character', 'Word')) +
    scale_x_discrete(labels = c('Twitter', 'Wikipedia')) +
    plot_theme + ylab('Optimal in Proportion of Runs') + xlab('')
ggsave('../paper/figures/preproc_token_type.png', 
       width = pe$p_width, height = 0.7*pe$p_width)
ggsave('../presentation/figures/preproc_token_type.png', 
       width = pe$p_width, height = 0.7*pe$p_width)


## Token type
## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
plot_dat = group_by(pp_dat, data_set, gram_size, token_type) %>%
    summarize(count = n()) %>%
    group_by(data_set, token_type) %>%
    mutate(prop = count / sum(count),
           gram_size = as.factor(gram_size))
plot_dat$token_type[plot_dat$token_type == 'char_wb'] = 'Token Type: Character'
plot_dat$token_type[plot_dat$token_type == 'word'] = 'Token Type: Word'

ggplot(plot_dat) +
    facet_wrap(~token_type, scales = 'free') +
    geom_bar(aes(x = data_set, y = prop, fill = gram_size), stat = 'identity',
             position = 'dodge', alpha = 0.8, color = 'white') +
    scale_fill_manual(values = pe$colors, name = 'Gram\nSize') +
    scale_x_discrete(labels = c('Twitter', 'Wikipedia')) +
    plot_theme + ylab('Optimal in Proportion of Runs') + xlab('')
ggsave('../paper/figures/preproc_gram_size.png', 
       width = pe$p_width, height = 0.7*pe$p_width)
ggsave('../presentation/figures/preproc_gram_size.png', 
       width = pe$p_width, height = 0.7*pe$p_width)

# ==============================================================================
# Relationship of f1-score and pre-processing choices
# ==============================================================================

## Token Type
ggplot(pp_dat) + 
    geom_boxplot(aes(x = data_set, y = f1, fill = token_type), 
                 alpha = 0.8, outlier.size = 0.1) +
    stat_summary(aes(x = data_set, y = f1, group = token_type), 
                 fun.y = mean, geom = 'point', shape = 3,
                 position = position_dodge(width = 0.75)) +
    scale_fill_manual(values = pe$colors, name = 'Token\nType',
                      labels = c('Character', 'Word')) +
    scale_color_manual(values = pe$colors, name = 'Token\nType',
                       labels = c('Character', 'Word')) +
    scale_x_discrete(labels = c('Twitter', 'Wikipedia')) +
    ylab('F1 Score') + xlab('') +
    plot_theme
ggsave('../paper/figures/preproc_f1_token_type.png', 
       width = pe$p_width, height = 0.7*pe$p_width)
ggsave('../presentation/figures/preproc_f1_token_type.png', 
       width = pe$p_width, height = 0.7*pe$p_width)

## Tfidf
ggplot(pp_dat) + 
    geom_boxplot(aes(x = data_set, y = f1, fill = tfidf), 
                 alpha = 0.8, outlier.size = 0.1) +
    stat_summary(aes(x = data_set, y = f1, group = tfidf), 
                 fun.y = mean, geom = 'point', shape = 3,
                 position = position_dodge(width = 0.75)) +
    scale_fill_manual(values = pe$colors, name = 'Token\nWeight',
                      labels = c('Count', 'Tfidf')) +
    scale_color_manual(values = pe$colors, name = 'Token\nWeight',
                       labels = c('Count', 'Tfidf')) +
    scale_x_discrete(labels = c('Twitter', 'Wikipedia')) +
    ylab('F1 Score') + xlab('') +
    plot_theme
ggsave('../paper/figures/preproc_f1_tfidf.png', 
       width = pe$p_width, height = 0.7*pe$p_width)
ggsave('../presentation/figures/preproc_f1_tfidf.png', 
       width = pe$p_width, height = 0.7*pe$p_width)

## Stemm/Lemma
ggplot(pp_dat) + 
    geom_boxplot(aes(x = data_set, y = f1, fill = stem), 
                 alpha = 0.8, outlier.size = 0.1) +
    stat_summary(aes(x = data_set, y = f1, group = stem), 
                 fun.y = mean, geom = 'point', shape = 3,
                 position = position_dodge(width = 0.75)) +
    scale_fill_manual(values = pe$colors, name = 'Normalize',
                      labels = c('Lemmatize', 'Stem')) +
    scale_color_manual(values = pe$colors, name = 'Normalize',
                       labels = c('Lemmatize', 'Stem')) +
    scale_x_discrete(labels = c('Twitter', 'Wikipedia')) +
    ylab('F1 Score') + xlab('') +
    plot_theme
ggsave('../paper/figures/preproc_f1_stem.png', 
       width = pe$p_width, height = 0.7*pe$p_width)
ggsave('../presentation/figures/preproc_f1_stem.png', 
       width = pe$p_width, height = 0.7*pe$p_width)

# ==============================================================================
# Plots exclusively for presentation
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
