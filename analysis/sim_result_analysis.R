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
# Height to width ratio for main results figures (f1, prec, rec)
htwr = 1
htwr_pres = 0.7

# ==============================================================================
# Data prep
# ==============================================================================
proc_file = function(filename, data_set) {
    cat(filename, '\n')
    comp = unlist(strsplit(filename, '/'))
    run = as.integer(comp[5])
    comp_2 = unlist(strsplit(comp[6], '_'))
    algo = comp_2[1]
    balance = as.numeric(gsub('\\.csv', '', comp_2[4]))
    if(is.element('icr', comp_2)) {
        icr = as.numeric(gsub('\\.csv', '', comp_2[6]))
    } else {
        icr = NA
    }
    # This file is corrupted somehow so I'll skip it 
    if(filename == '../data/runs/wikipedia_hate_speech/10/random_simulation_data_0.10.csv') {
        return(NA)
    }
    data = suppressMessages(read_csv(filename)) %>%
        mutate(algo = algo, balance = balance, run = run, icr = icr,
               total_samples = proj_conf$data_sets[[data_set]]$n_docs)       
    # TODO: There are inconsistent columns in the breitbart results
    if("clf__tol" %in% colnames(data)) {
        data = select(data, -clf__tol)    
    }
    return(data)
}

# TODO: If results for different query algorithms come in separate files, this 
# code has to be adapted to put them all in the same DF. Adjust algo in 
# proc_file() accordingly
results = list()
i = 1
for(data_set in DATA_SETS) {
    
    cat('Processing ', data_set, '\n')    
    
    inpath = paste0(DATA_DIR, data_set) 
    files = list.files(inpath, recursive = TRUE, pattern = '\\d.csv$')
    files = paste0(inpath, '/', files)
    
    dfs = lapply(files, proc_file, data_set) 
    data = do.call(rbind, dfs) %>%
        mutate(f1_kcore = ifelse(is.na(f1), 0, f1),
               precision = ifelse(is.na(p), 0, p),
               recall = ifelse(is.na(r), 0, r),
               balance = paste0("Balance: ", balance),
               data_set = data_set)
    results[[i]] = data
    i = i + 1
}   
#save(results, file = 'results_cache.RData')

#load('results_cache.RData')
data = do.call(rbind, results)
data$data_set = recode(data$data_set, 'tweets' = 'Twitter', 
                      'wikipedia_hate_speech' = 'Wikipedia',
                      'breitbart' = 'Breitbart')
icr_data = filter(data, !is.na(icr))
data = filter(data, balance %in% c("Balance: 0.01", "Balance: 0.1", 
                                   "Balance: 0.3", "Balance: 0.5"))

n_dset = length(unique(data$data_set))

# separate out the intercoder reliability runs
data = filter(data, is.na(icr)) %>% select(-icr)

# ==============================================================================
# Visualize results
# ==============================================================================

# F1 score
data = filter(data, batch * BATCH_SIZE <= 4000)
ggplot(data, aes(x = batch * BATCH_SIZE #/ total_samples * 100
                 , y = f1, 
                 color = algo, linetype = algo)) +
    facet_wrap(~balance + data_set, scales = 'free', ncol = n_dset) +
    geom_point(size = 0.05, alpha = 0.05) + 
    geom_smooth() +
    scale_color_manual(values = pe$colors, name = 'Labeling\nAlgorithm') +
    scale_linetype(name = 'Labeling\nAlgorithm') +
    ylab('F1 Score') + 
    xlab('# of labeled samples') + 
    #xlab('% of samples labeled') +
    plot_theme
ggsave('../paper/figures/main_results_f1.png', width = pe$p_width, 
       height = htwr*pe$p_width)
ggsave('../presentation/figures/main_results_f1.png', width = pe$p_width, 
       height = htwr_pres*pe$p_width)

# Precision
ggplot(data, aes(x = batch * BATCH_SIZE #/ total_samples * 100
                 , 
                 y = precision, color = algo, linetype = algo)) +
    facet_wrap(~balance + data_set, scales = 'free', ncol = n_dset) +
    geom_point(size = 0.05, alpha = 0.05) + 
    geom_smooth() +
    scale_color_manual(values = pe$colors, name = 'Labeling\nAlgorithm') +
    scale_linetype(name = 'Labeling\nAlgorithm') +
    ylab('Precision') + xlab('# labeled samples') +
    plot_theme
ggsave('../paper/figures/main_results_precision.png', width = pe$p_width, 
       height = htwr*pe$p_width)
ggsave('../presentation/figures/main_results_precision.png', 
       width = pe$p_width, height = htwr_pres*pe$p_width)

# Recall
ggplot(data, aes(x = batch * BATCH_SIZE #/ total_samples * 100
                 , 
                 y = recall, color = algo, linetype = algo)) +
    facet_wrap(~balance + data_set, scales = 'free', ncol = n_dset) +
    geom_point(size = 0.05, alpha = 0.05) + 
    geom_smooth() +
    scale_color_manual(values = pe$colors, name = 'Labeling\nAlgorithm') +
    scale_linetype(name = 'Labeling\nAlgorithm') +
    ylab('Recall') + xlab('# labeled samples') +
    plot_theme
ggsave('../paper/figures/main_results_recall.png', 
       width = pe$p_width, height = htwr*pe$p_width)
ggsave('../presentation/figures/main_results_recall.png', 
       width = pe$p_width, height = htwr_pres*pe$p_width)

# Visualize support growth
data = do.call(rbind, results)
data$data_set = recode(data$data_set, 'tweets' = 'Twitter', 
                      'wikipedia_hate_speech' = 'Wikipedia',
                      'breitbart' = 'Breitbart')
d = filter(data, balance == 'Balance: 0.05') %>%
    group_by(batch, algo, data_set) %>%
    summarize(mean_f1 = mean(f1_kcore), mean_support = mean(support))
#total_positives = proj_conf$data_sets$tweets$n_positive

ggplot(d, aes(x = batch * BATCH_SIZE #/ total_samples * 100
              , 
              y = mean_f1, color = algo, size = mean_support)) + 
    facet_wrap(~data_set, ncol = 1) +
    geom_point(alpha = 0.4) + 
    #geom_line(size = 1) +
    scale_color_manual(values = pe$colors, name = 'Labeling\nAlgorithm') +
    scale_linetype(name = 'Labeling\nAlgorithm') +
    scale_size_continuous(name = 'Mean # of\npositives', range = c(0.5, 4)) +
    ylab('F1-Score (mean)') + xlab('# labeled samples') +
    plot_theme
ggsave('../paper/figures/f1_labeled_support_balance_001.png', 
       width = pe$p_width, height = 0.7*pe$p_width)
ggsave('../presentation/figures/f1_labeled_support_balance_001.png', 
   width = pe$p_width, height = 0.7*pe$p_width)


# ==============================================================================
# Pre-pocessing choices
# ==============================================================================

# Parse out all preprocessing options from text selector argument    
pp_dat = filter(data, !is.na(text__selector__fname))
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
    #scale_x_discrete(labels = c('Twitter', 'Wikipedia')) +
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
    #scale_x_discrete(labels = c('Twitter', 'Wikipedia')) +
    plot_theme + ylab('Optimal in Proportion of Runs') + xlab('')
ggsave('../paper/figures/preproc_token_type.png', 
       width = pe$p_width, height = 0.7*pe$p_width)
ggsave('../presentation/figures/preproc_token_type.png', 
       width = pe$p_width, height = 0.7*pe$p_width)


## Gram Size
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
    #scale_x_discrete(labels = c('Twitter', 'Wikipedia')) +
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
# Intercoder reliability plots
# ==============================================================================
icr_data = filter(icr_data, balance %in% c("Balance: 0.05", "Balance: 0.1"))
n_icr = length(unique(icr_data$icr))
ggplot(icr_data, aes(x = batch * BATCH_SIZE, y = f1, color = algo,
                 linetype = algo)) +
    facet_wrap(~balance + icr, scales = 'free', ncol = n_icr) +
    geom_point(size = 0.05, alpha = 0.05) +
    geom_smooth() +
    scale_color_manual(values = pe$colors, name = 'Labeling\nAlgorithm') +
    scale_linetype(name = 'Labeling\nAlgorithm') +
    ylab('F1 Score') + xlab('# labeled samples') +
    plot_theme
ggsave('../paper/figures/icr_results_f1.png', width = pe$p_width, 
       height = htwr*pe$p_width)
ggsave('../presentation/figures/icr_results_f1.png', width = pe$p_width, 
       height = htwr_pres*pe$p_width)

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
