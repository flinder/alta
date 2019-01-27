#devtools::install_github('tidyverse/dplyr')
library(dplyr) # Note that this is currently the development version which 
               # contains a new feature to keep empty groups in 
               # group_by() - summarize() operations (used for the results table)
library(ggplot2)
#devtools::install_github('flinder/flindR')
library(flindR)
library(reshape2)
library(yaml)
library(stringr)
library(xtable)
library(tidyr)
library(readr)
library(doParallel)

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

# This function parses experiment result files (see github readme for filename
# conventions).
parse_filename = function(path) {
    # Get the run idx from the directory name
    path_components = unlist(strsplit(path, '/'))
    run = as.integer(path_components[5])
    data_set = path_components[4]
    
    # Parse the experimental conditions from the filename 
    filename = path_components[6]
    components = unlist(strsplit(gsub('.csv', '', filename), '_'))
    
    ## Check which of the various simulations it is and parse corespondingly
    # Cols:
    # - experiment_name
    # - mode (active/random)
    # - query_algo
    # - balance
    # - icr (NA for non icr experiments)
    # - prop_random
    # - )
    mode = components[1]
    if(length(components) == 4) {
        experiment_name = 'main_old'
        if(mode == 'random') query_algo = ''
        else query_algo = 'margin'
        balance = as.numeric(components[4])
        icr = NA
        prop_random = NA
    } else if(length(components) == 5) {
        experiment_name = 'main_new'
        query_algo = components[4]
        balance = as.numeric(components[5])
        icr = NA
        prop_random = NA
        if(mode == 'random') {
            # This is only the case for Breitbart where we had to re-run random
            # too because the dataset changed
            query_algo = ''
        }
    } else if(length(components) == 6) {
        experiment_name = 'icr_old'
        if(mode != 'random') query_algo = 'margin'
        else query_algo = ''
        balance = as.numeric(components[4])
        icr = as.numeric(components[6])
        prop_random = NA
    } else if(length(components) == 7) {
        experiment_name = 'icr_proprand'
        query_algo = components[4]
        balance = as.numeric(components[5])
        if(components[6] == 'icr') {
            icr = as.numeric(components[7])
            prop_random = NA
        } else if(components[6] == 'rand') {
            icr = NA
            prop_random = as.numeric(components[7])
        } else {
            stop('Unexpected filename structure')
        }
    } else {
        return(list('experiment_name' = NA, 
                'run' = NA, 
                'mode' = NA, 
                'query_algo' = NA, 
                'balance' = NA, 
                'icr' = NA, 
                'prop_random' = NA, 
                'filename' = filename,
                'data_set' = NA))
            
    }
    return(list('experiment_name' = experiment_name, 
                'run' = run, 
                'mode' = mode, 
                'query_algo' = query_algo, 
                'balance' = balance, 
                'icr' = icr, 
                'prop_random' = prop_random, 
                'filename' = filename,
                'data_set' = data_set))
}

col_or_na = function(df, col) {
    if(col %in% colnames(df)) return(as.data.frame(df)[, col])
    else return(NA)
}


# This function extracts metadata from filename, loads the results of the
# file and returns a standardized dataframe with all information
# Returned variables:
# - Experimental metadata (see above)
# - precision (validation / training)
# - recall (validation / training)
# - f1 score (validation / training)
# - preprocessing string
# - batch
# - support
# - proportion random
# - balance
# - algorithm (random, active)
# - query algorithm
process_file = function(path) {
    cat(path, '\n')
    experiment = parse_filename(path)
    result_data = try(suppressMessages(read_csv(path)))
    if(inherits(result_data, 'try-error')) {
        cat('Error. Skipping.\n') 
        return(NULL) 
    } 
    if(nrow(result_data) == 0) {
        print('No data')
        return(NULL) 
    } 
    out_data = data_frame(
        data_set = experiment$data_set,
        experiment_name = experiment$experiment_name,
        f1_validation = result_data$f1,
        precision_validation = result_data$p,
        recall_validation = result_data$r,
        f1_training = col_or_na(result_data, 'f1_l'),
        precision_training = col_or_na(result_data, 'p_l'),
        recall_training = col_or_na(result_data, 'r_l'),
        preprocessing_string = result_data$text__selector__fname,
        batch = result_data$batch,
        support = result_data$support,
        balance = experiment$balance,
        mode = experiment$mode,
        query_algorithm = experiment$query_algo,
        run = experiment$run,
        icr = experiment$icr,
        prop_random = experiment$prop_random
        )
    return(out_data)
}


files = list.files(DATA_DIR, recursive = TRUE, pattern = '\\d.csv$')

# This file causes memory error:
files = paste0(DATA_DIR, files)
registerDoParallel(cores=8)
dfs = foreach(i=1:length(files)) %dopar% process_file(files[i])

data = bind_rows(dfs) %>%
    filter(!is.na(experiment_name)) %>%
    mutate(f1_validation = ifelse(is.na(f1_validation), 0, f1_validation),
           precision_validation = ifelse(is.na(precision_validation), 0, 
                                         precision_validation),
           recall_validation = ifelse(is.na(recall_validation), 0, 
                                      recall_validation),
           balance = paste0("Balance: ", balance))
# Remove the zeros
#data = filter(data, f1_validation != 0)

data$data_set = recode(data$data_set, 'tweets' = 'Twitter', 
                      'wikipedia_hate_speech' = 'Wikipedia',
                      'breitbart' = 'Breitbart')

# ==============================================================================
# Visualize main experiment results
# ==============================================================================

# Select the following:
## Twitter: 
##    - passive learning results (random) from the old experiment
##    - active learning results (margin) from the icr_proprand experiment wher
##      proportion random is 0
##    - active learning results (committee / emc) from the main new experiment
## Wiki:
##    - passive learning results (random) from the old experiment
##    - active learning results (committee / margin ) from new experiment
## Breitbart
##    - passive and active learning results (committee / margin) from new exp.

main_pdat = data
main_pdat$selected = 0

## All results from new experiment. This is also for wiki and breitbart
main_pdat$selected[main_pdat$experiment_name == 'main_new'] = 1
# Old random results for Twitter and Wiki
main_pdat$selected[(main_pdat$data_set %in% c('Twitter', 'Wikipedia') & 
               main_pdat$mode == 'random' & 
               main_pdat$experiment_name == 'main_old')] = 1
# The margin active learning form the icr_proprand experiment (prop_random 0) 
main_pdat$selected[(main_pdat$data_set == 'Twitter' & 
               main_pdat$mode == 'active' & 
               main_pdat$experiment_name == 'icr_proprand' & 
               main_pdat$prop_random == 0)] = 1
main_pdat = filter(main_pdat, selected == 1,
              balance %in% c("Balance: 0.01", "Balance: 0.05", 
                             "Balance: 0.1", "Balance: 0.5"),
              (batch + 1) * BATCH_SIZE <= 5000) %>%
    mutate(algorithm = paste0(mode, '_', query_algorithm))

# F1 score
ggplot(main_pdat, aes(x = (batch + 1) * BATCH_SIZE, 
                 y = f1_validation, 
                 color = algorithm, 
                 linetype = algorithm)) +
    facet_wrap(~balance + data_set, scales = 'free', ncol = 3) +
    #geom_point(size = 0.05, alpha = 0.05) + 
    geom_smooth(method = 'loess') +
    #geom_smooth() +
    scale_color_manual(values = pe$colors[c(4:1)], name = 'Labeling\nAlgorithm') +
    scale_linetype(name = 'Labeling\nAlgorithm') +
    ylim(0, 1) +
    ylab('F1 Score') + 
    xlab('# of labeled samples') + 
    #xlab('% of samples labeled') +
    plot_theme
ggsave('../paper/figures/main_results_f1.png', width = pe$p_width, 
       height = htwr*pe$p_width, dpi = 100)
ggsave('../presentation/figures/main_results_f1.png', width = pe$p_width, 
       height = htwr_pres*pe$p_width, dpi = 100)

# Table: Average amount of training data required to reach levels of f1 score
empty_robust_ci = function(x) {
    out = try(t.test(x), silent = TRUE)
    if(inherits(out, 'try-error')) {
        return(c(NA, NA))
    } else {
        return(out$conf.int)
    }
}

# Make table to interpret the ratio of required training data
main_pdat %>% 
    filter(query_algorithm %in% c('margin', '')) %>%
    mutate(n_training_samples = (batch+1) * BATCH_SIZE,
           algorithm = ifelse(mode == 'random', 'random', 'active')) %>%
    select(run, batch, n_training_samples, f1_validation, algorithm, 
           data_set, balance) %>%
    group_by(run, algorithm, data_set, balance) %>%
    mutate(first_01 = cumsum(cumsum(f1_validation >= 0.1)) == 1,
           first_05 = cumsum(cumsum(f1_validation >= 0.5)) == 1,
           first_08 = cumsum(cumsum(f1_validation >= 0.8)) == 1) %>%
    melt(id.vars = c('run', 'batch', 'n_training_samples', 'f1_validation', 
                     'algorithm', 'data_set', 'balance')) %>%
    tbl_df() %>%
    mutate(value = factor(value, levels = c('TRUE', 'FALSE'))) %>%
    group_by(algorithm, variable, value, data_set, balance) %>%
    summarize(average_td = mean(n_training_samples)) %>%
    ungroup() %>%
    filter(value == 'TRUE') %>%
    select(-value) %>% 
    spread(algorithm, average_td) %>%
    mutate(ratio = random / active,
           f1_score = as.numeric(sapply(strsplit(as.character(variable), '_'), 
                                        function(x) x[2])) / 10,
           balance = as.numeric(sapply(strsplit(balance, ' '), 
                                       function(x) x[2]))) %>%
    select(-variable) %>%
    arrange(data_set, balance) %>%
    filter(f1_score == 0.1) %>%
    rename('Data Set' = data_set, 'Balance' = balance, 
           '# Samples Active' = active, '# Samples Passive' = random,
           'Ratio' = ratio) %>%
    select(-f1_score) %>%
    xtable(caption = 'Number of training samples required to reach an F1-Score of 0.1 for active and passive learning.',
           label = 'tab:n_train_samples') %>%
    print(file = '../paper/tables/n_train_samples.tex', 
          include.rownames = FALSE)

# Precision
ggplot(main_pdat, aes(x = (batch + 1) * BATCH_SIZE, 
                 y = precision_validation, color = algorithm, 
                 linetype = algorithm)) +
    facet_wrap(~balance + data_set, scales = 'free', ncol = 3) +
    #geom_point(size = 0.05, alpha = 0.05) + 
    geom_smooth(method = 'loess') +
    #geom_smooth() +
    scale_color_manual(values = pe$colors[c(4:1)], name = 'Labeling\nAlgorithm') +
    scale_linetype(name = 'Labeling\nAlgorithm') +
    ylab('Precision') + xlab('Labeled samples') +
    plot_theme
ggsave('../paper/figures/main_results_precision.png', width = pe$p_width, 
       height = htwr*pe$p_width, dpi = 100)
ggsave('../presentation/figures/main_results_precision.png', 
       width = pe$p_width, height = htwr_pres*pe$p_width, dpi = 100)

# Recall
ggplot(main_pdat, aes(x =  (batch + 1) * BATCH_SIZE, 
                 y = recall_validation, color = algorithm, 
                 linetype = algorithm)) +
    facet_wrap(~balance + data_set, scales = 'free', ncol = 3) +
    #geom_point(size = 0.05, alpha = 0.05) + 
    geom_smooth(method = 'loess') +
    #geom_smooth() +
    scale_color_manual(values = pe$colors[c(4:1)], name = 'Labeling\nAlgorithm') +
    scale_linetype(name = 'Labeling\nAlgorithm') +
    ylab('Recall') + xlab('Labeled samples') +
    plot_theme
ggsave('../paper/figures/main_results_recall.png', 
       width = pe$p_width, height = htwr*pe$p_width, dpi = 100)
ggsave('../presentation/figures/main_results_recall.png', 
       width = pe$p_width, height = htwr_pres*pe$p_width, dpi = 100)

# Visualize support growth
sg_pdat = filter(main_pdat,
                 algorithm %in% c('active_margin', 'random_'),
                 balance == 'Balance: 0.01') %>%
    group_by(batch, algorithm, data_set) %>%
    summarize(mean_f1 = mean(f1_validation), 
              mean_support = mean(support))

#total_positives = proj_conf$data_sets$tweets$n_positive
ggplot(sg_pdat, aes(x = (batch + 1) * BATCH_SIZE, y = mean_f1, 
                    color = algorithm, size = mean_support)) + 
    facet_wrap(~data_set, ncol = 1, scales = 'free') +
    geom_point(alpha = 0.4) +
    #geom_line(size = 1) +
    scale_color_manual(values = pe$colors[c(2,1)], name = 'Labeling\nAlgorithm') +
    scale_linetype(name = 'Labeling\nAlgorithm') +
    scale_size_continuous(name = 'Mean # of\npositives', range = c(0.5, 4)) +
    ylab('F1-Score (mean)') + xlab('Labeled samples') +
    plot_theme
ggsave('../paper/figures/f1_labeled_support_balance_001.png', 
       width = pe$p_width, height = 0.7*pe$p_width, dpi = 100)
ggsave('../presentation/figures/f1_labeled_support_balance_001.png', 
   width = pe$p_width, height = 0.7*pe$p_width, dpi = 100)

# ==============================================================================
# Intercoder reliability plots
# ==============================================================================
icr_data = filter(data, experiment_name %in% c('icr_proprand', 'icr_old'),
                  balance %in% c("Balance: 0.05", "Balance: 0.1"),
                  !is.na(icr),
                  !(experiment_name == 'icr_old' & query_algorithm == 'margin')) %>%
    mutate(algorithm = paste0(mode, '_', query_algorithm))#,
           #exper = paste0(algorithm, '_', experiment_name))
icr_levels = unique(icr_data$icr)
n_icr = length(icr_levels)

for(l in icr_levels) {
    icr_data$icr[icr_data$icr == l] = paste0('Reliability: ', l)
}

ggplot(icr_data, aes(x = (batch + 1) * BATCH_SIZE, y = f1_validation, 
                     color = algorithm, linetype = algorithm)) +
    facet_wrap(~balance + icr, scales = 'free', ncol = n_icr) +
    #geom_point(size = 0.05, alpha = 0.05) +
    geom_smooth(method = 'loess') +
    #geom_smooth() +
    scale_color_manual(values = pe$colors[c(4, 2, 1)], 
                       name = 'Labeling\nAlgorithm') +
    scale_linetype(name = 'Labeling\nAlgorithm') +
    ylab('F1 Score') + xlab('Labeled samples') +
    ylim(0, 1) +
    plot_theme
ggsave('../paper/figures/icr_results_f1.png', width = 1.3 * pe$p_width, 
       height = 0.8 * htwr*pe$p_width)
ggsave('../presentation/figures/icr_results_f1.png', width = 1.3 * pe$p_width, 
       height = 0.8 * htwr_pres*pe$p_width)

# ==============================================================================
# Generalization Error Plots
# ==============================================================================

# For this we compare the training and validation loss 
ge_pdat = filter(main_pdat,
                 algorithm == 'active_margin') %>%
    select(f1_validation, f1_training, data_set, batch, balance, 
           algorithm) %>%
    gather(key = 'score_type', value = 'f1_score', -data_set:-algorithm)

ggplot(ge_pdat, aes(x = (batch + 1) * BATCH_SIZE, y = 1-f1_score, 
                    color = score_type, linetype = score_type)) +
    #geom_smooth(method = 'loess') +
    geom_smooth() +
    facet_wrap(~balance + data_set, scales = 'free') + 
    scale_color_manual(values = pe$colors[c(6, 7)], 
                       name = 'Score\nType', 
                       labels = c('Training', 'Validation')) + 
    scale_linetype_manual(values = c(1, 2), 
                          name = 'Score\nType',
                          labels = c('Training', 'Validation')) + 
    ylim(0, 1) +
    ylab('1 - F1 Score') + xlab('Labeled samples') +
    pe$theme
ggsave('../paper/figures/generalization_error.png', width = pe$p_width, 
       height = htwr*pe$p_width, dpi = 100)

# ==============================================================================
# Proportion Random
# ==============================================================================

pr_pdat = filter(data, experiment_name == 'icr_proprand',
                 !is.na(prop_random),
                 balance %in% c("Balance: 0.01", "Balance: 0.05", 
                                "Balance: 0.1", "Balance: 0.5"),
                 prop_random != 0.2) %>%
    mutate(prop_random = as.factor(prop_random))

ggplot(pr_pdat, aes(x = (batch + 1) * BATCH_SIZE, y = f1_validation, color = prop_random,
           linetype = prop_random)) +
    #geom_point(size = 0.5, alpha = 0.1) +
    facet_wrap(~balance, scales = 'free') +
    geom_smooth(method = 'loess') +
    #geom_smooth() +
    scale_color_manual(values = pe$colors, 
                       name = 'Proportion\nRandom') + 
    scale_linetype_manual(values = 1:4, 
                          name = 'Proportion\nRandom') + 
    ylab('F1 Score') + xlab('Labeled samples') +
    ylim(0, 1) +
    pe$theme
ggsave('../paper/figures/proportion_random.png', width = pe$p_width, 
       height = htwr*pe$p_width, dpi = 100)

# ==============================================================================
# Pre-pocessing choices
# ==============================================================================

# Parse out all preprocessing options from text selector argument    
pp_dat = filter(data, !is.na(preprocessing_string))
pp_info = pp_dat$preprocessing_string
pp_dat$token_type = gsub("'", '', str_extract(pp_info, "'.+'"))
pp_dat$gram_size = as.integer(gsub("'", '', str_extract(pp_info, "\\d")))
weight_stem_info = str_extract(pp_info, '(True|False)_(True|False)')
tfidf_str = sapply(weight_stem_info, function(x) unlist(strsplit(x, '_'))[1])  
pp_dat$tfidf = tfidf_str == 'True'
stem_str = sapply(weight_stem_info, function(x) unlist(strsplit(x, '_'))[2])  
pp_dat$stem = stem_str == 'True'
    
pp_dat = select(pp_dat, -preprocessing_string)  

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
