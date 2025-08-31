
library(pheatmap)
library(RColorBrewer)
library(cowplot)
library(ggplot2)

len <- length
printf <- function(...) {print(sprintf(...))}
ph <- function(...) {pheatmap(..., cluster_rows=F, cluster_cols=F)}
spec_cg <- colorRampPalette(rev(brewer.pal(n = 11,  name = "Spectral")))
rdylbu_cg <- colorRampPalette(rev(brewer.pal(n = 11,  name = "RdYlBu")))

base_path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/results/gfp_dataset/llm_epinnet_comparisions/"

figure_path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/figures/esm_lr_epinnet/"
dir.create(figure_path)
datasets = list("train_1.csv", "train_2.csv", "train_3.csv", "train_4.csv", "train_5.csv")




plot_boxplots <- function(df, metric)  {
  melted_df <- melt(df)
  colnames(melted_df) <- c("nmuts", "model_name", "vals")
  melted_df <- melted_df[!is.na(melted_df$vals),]
  
  ggplot(melted_df, aes(x=model_name, y=vals)) +
    geom_boxplot() +
    geom_dotplot(binaxis = "y", stackdir = "center", position = "dodge")
}

evaluate_classifier <- function(gt, pred, pred_prob) {
  

  
  confusion_matrix <- matrix(0, nrow=2, ncol=2)
  
  # TP: Positive predicted as positive
  confusion_matrix[1,1] <- sum((gt == 0) & (pred == 0))
  
  # FN: Positive predicted as negative
  confusion_matrix[1,2] <- sum((gt == 0) & (pred == 1))
  
  # FP: Negative predicted as positive
  confusion_matrix[2,1] <- sum((gt == 1) & (pred == 0))
  
  # TN: Negative predicted as negative
  confusion_matrix[2,2] <- sum((gt == 1) & (pred == 1))
  
  accuracy = (confusion_matrix[1,1] + confusion_matrix[2,2]) / sum(confusion_matrix)
  precision = (confusion_matrix[1,1]) / (confusion_matrix[1,1] + confusion_matrix[2,1])
  recall = (confusion_matrix[1,1]) /  (confusion_matrix[1,1] + confusion_matrix[1,2])
  f1 = 2*(precision*recall) / (precision+recall)
  
  accuracy = ifelse(is.nan(accuracy), 0, accuracy)
  precision = ifelse(is.nan(precision), 0, precision)
  recall = ifelse(is.nan(recall), 0, recall)
  f1 = ifelse(is.nan(f1), 0, f1)
  
  
  
  FPR_bins = seq(0, 1, length.out=200)
  
  positive_ecdf = ecdf(pred_prob[gt==0])
  positive_percentiles = quantile(pred_prob[gt==0], FPR_bins)
  negative_ecdf = ecdf(pred_prob[gt==1])
  negative_percentiles = quantile(pred_prob[gt==1], FPR_bins)
  
  
  # if you're reading this code, I decided instead of using calc_roc stock function
  # to force myself to calculate it on my own, just to show myself how it works and why it's even an important measurement
  # :)
  ROC = sum(positive_ecdf(rev(negative_percentiles)) * 1/len(FPR_bins))
  
  plot(rev(FPR_bins), positive_ecdf(rev(negative_percentiles)), type="l", xlim=c(0,1), ylim=c(0,1))
  lines(c(0,1), c(0,1), lty=2)
  
  results = list("confusion_matrix"= confusion_matrix,
                 "accuracy" = accuracy,
                 "precision"=precision,
                 "recall"=recall,
                 "f1"=f1,
                 "ROC"=ROC,
                 "top_100"=(sum(gt[order(pred_prob)][1:100] == 0)/100))
}



train_nmuts = 1:5
max_test_nmuts <- 11


model_cols = list("esm8m"=list("gt"="y_true_llm_esm2_8m", 
                               "pred"="y_pred_llm_esm2_8m_mlp_label",
                               "pred_score"="y_pred_llm_esm2_8m_mlp_score",
                               "nmuts"="nmuts_llm_esm2_8m"),
                  "esm35m"=list("gt"="y_true_llm_esm2_35m", 
                                "pred"="y_pred_llm_esm2_35m_mlp_label",
                                "pred_score"="y_pred_llm_esm2_35m_mlp_score",
                                "nmuts"="nmuts_llm_esm2_35m"),
                  
                  "esm650m"=list("gt"="y_true_llm_esm2_650m", 
                                 "pred"="y_pred_llm_esm2_650m_mlp_label",
                                 "pred_score"="y_pred_llm_esm2_650m_mlp_score",
                                 "nmuts"="nmuts_llm_esm2_650m"),              
                  
                  "lr"=list("gt"="y_true_seq", 
                            "pred"="y_pred_ohe_linreg_label",
                            "pred_score"="y_pred_ohe_linreg",
                            "nmuts"="nmuts_seq"),
                  
                  "epinnet"=list("gt"="y_true_seq", 
                                 "pred"="y_pred_ohe_mlp_label",
                                 "pred_score"="y_pred_ohe_mlp_score",
                                 "nmuts"="nmuts_seq"))

models = c("esm8m", "esm35m", "esm650m", 'lr', 'epinnet')
metrics <- c("accuracy", "precision", "recall", "f1", "ROC", "top_100")
evaluation_across_all <- list()
specific_evaluations <- list()


for (metric in metrics) {
  evaluation_across_all[[metric]] = matrix(NaN, nrow=len(train_nmuts), ncol=len(models))
  colnames(evaluation_across_all[[metric]]) <- models
  rownames(evaluation_across_all[[metric]]) <- train_nmuts
}


for (model in models) {
  specific_evaluations[[model]] <- list()
  
  for (metric in metrics) {
    specific_evaluations[[model]][[metric]] <- matrix(NaN, nrow=len(train_nmuts), ncol=max_test_nmuts)
  }
}

for (train_nmut in train_nmuts) {
  df = read.csv(sprintf("%s/train_%d.csv", base_path, train_nmut))
  
  
  printf("Evaluation for %d mutations", train_nmut)
  
  
  for (model in models) {
    printf("    Evaluating for %s", model)
    eval = evaluate_classifier(df[[model_cols[[model]]$gt]], 
                               df[[model_cols[[model]]$pred]],
                               df[[model_cols[[model]]$pred_score]])
    
    for (metric in metrics) {
      evaluation_across_all[[metric]][train_nmut,model] <- eval[[metric]]
    }
  }
  
  for (test_nmut in (train_nmut + 1):max_test_nmuts) {
    printf("    Evaluating train on %d test on %d", train_nmut, test_nmut)
    # sanity 
    
    subset_ind <- rep(T, nrow(df))
    
    print("Sanity to make sure all models had the same indices = the subsetting should be identical for all")
    for (model in models) {
      subset_ind = subset_ind & df[[model_cols[[model]]$nmuts]] == test_nmut
      print(sum(subset_ind))
    }
    printf("Sanity all the above %d numbers should've been the same", len(models))
    
    subset_df = df[subset_ind,]
    
    for (model in models) {
      printf("        Evaluating for %s[%d->%d]", model, train_nmut, test_nmut)
      
      specific_eval = evaluate_classifier(subset_df[[model_cols[[model]]$gt]], 
                                          subset_df[[model_cols[[model]]$pred]],
                                          subset_df[[model_cols[[model]]$pred_score]])
      
      for (metric in metrics) {
        specific_evaluations[[model]][[metric]][train_nmut,test_nmut] <- specific_eval[[metric]]
      }
    }
  }
}


plot_list <- list()

for (model_name in models) {
  for (metric in metrics) {
    
    specific_evaluations[[model_name]][[metric]]
    
    mt <- specific_evaluations[[model_name]][[metric]]
    colnames(mt) <- c(1:11)
    rownames(mt) <- c(1:5)
    
    
    text_mat = matrix("", nrow=nrow(mt),
                          ncol=ncol(mt))
    
    for (i in 1:nrow(mt)) {
      for (j in 1:ncol(mt)) {
        if (!is.na(mt[i,j])) {
          round_number = round(mt[i,j], digits=2) * 100
          if (round_number == 0) {
            text_mat[i,j] <- "0"
          } else if (round_number == 100) {
            text_mat[i,j] <- "1"
          }
          else if (round_number < 10) {
            text_mat[i,j] <- sprintf(".0%g", round_number)
          } else {
            text_mat[i,j] <- sprintf(".%g", round_number)  
          }
          
        }
      }
    }
    
    plt <- ph(mt, 
              #display_numbers=text_mat,
              main=sprintf("%s - %s", model_name, metric),
              #breaks=seq(0,1,length=500),
              #col=spec_cg(500),
              legend=T,
              number_color="black")[[4]]
    
    
    plot_list <- append(plot_list, list(plt))
    
  }
}


plot_list$nrow <- len(models)
final_plot <- do.call(plot_grid, plot_list)

pdf(sprintf("%s/specific_comp_val.pdf", figure_path), width=13, height=6.5)
plot(final_plot)
dev.off()





