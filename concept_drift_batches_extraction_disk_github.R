require(ggpubr)
require(effsize)
require(xtable)
require(ScottKnott)
require(gtools)
require(stringi)
require(stringr)
require(scales)
require(tidyr)
require(ggplot2)
require(dplyr)

install.packages('gtools', repos='http://cran.us.r-project.org')
install.packages("dplyr")
install.packages("ggplot2")

library(dplyr)
library(ggplot2)

datasets <- c('disk')
named_data <- c('Disk_concept_drift_in_batches')
names(named_data) <- datasets

#Change models accordingly
dataset <- 'disk'
models <- c('rf')
df <- NA
dfi <- NA

# read csv generated from the python script
#dfj <- read.csv(paste('./results/', paste('concept_drift', dataset, 'rf', 'feature_importance_rs1', sep='_'), '.csv', sep=''))
dfj <- read.csv(paste('./Documents/phd_related/AIOps_disk_failure_prediction/feature_importance/', paste('concept_drift', dataset, '2015_rf_week', 'feature_importance_rs1', sep='_'), '.csv', sep=''))


# create a vector to store pvalues
pvals <- numeric(nrow(dfj))

# calculate pvalues
for (i in 1:nrow(dfj)) {
  res <- prop.test(c(dfj[i, 3], dfj[i, 4]), c(dfj[i, 5], dfj[i, 6]), p = NULL, alternative = "two.sided", correct = TRUE)
  pvals[i] <- res$p.value
}

# create dataframe with differences between error rates
dfj$diff = abs((dfj$Testing_Error_Rate/dfj$Testing_Size) - (dfj$Training_Error_Rate/dfj$Training_Size)) / (dfj$Training_Error_Rate/dfj$Training_Size)

# create datasets with final results and attribute a TRUE (it is drift) when pvals is smaller than 0.05 and FALSE otherwhise
dfi <- data.frame(X=factor(2:(nrow(dfj)/length(models)+1)), Sig=(pvals < 0.05), Y=dfj$diff, P=dfj$Weeks, FI=dfj$Feature_Importance, Dataset = 'Google', Model = 'Random Forests')


# plot the batches with drift and the drift severity according to the relative difference in error rate
ggplot(df %>% filter(Dataset=='Backblaze'), aes(x=X, y=Y, shape=Sig, color=Sig)) + geom_point() + 
  geom_hline(yintercept=0) + 
  facet_grid(.~Model, scales='free_x') +
  labs(x=substitute(paste(bold('Time Period'))), y=substitute(paste(bold('Relative difference of error rate'))))  + scale_color_discrete(name='Batch Label') + 
  scale_shape_manual(name='Batch Label', values=c(19, 15, 17)) + 
  scale_x_discrete(breaks=seq(2, 51, 2))
ggsave('concept_drift_disk.pdf', width=190, height=60, units='mm')

# write in file new csv with ground truth results
write.csv(dfi, file = './results/rf_concept_drift_localization_disk_r_1.csv')
