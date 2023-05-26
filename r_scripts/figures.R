###
##  Figures: paper multi-target regression
###

rm(list = ls())
library(tidyverse)
library(RColorBrewer)
library(ggpubr)

source(
  paste0(
    "https://raw.githubusercontent.com/anaguilarar/R_general_functions/master/scripts//", 
    "plot_functions.R"
  )
)


# outputpaths
modelresultspath = "D:/OneDrive - Universidad Nacional de Colombia/PhD/Hyper_Spec_Camera/paper/results/"
STRESULTSPATH = "st_results.csv"
MTRESULTSPATH = "mt_results.csv"
TESTRESULTSPATH = "test_results.csv"
PLOTS_OUTPUTPATH = "plots/"

# variables names
MODELNAMES = c("Lasso","PLS","RF","Ridge","SVM-Linear")
MODELS = c("lasso","pls","rf","ridge","svr_linear")

# elements order

idlabels = c("Li","B","Na","Mg","P","S","K",
             "Ca","Mn","Fe","Co","Cu","Zn","Rb","Sr","Mo","Cd")

###### Single target results
dataevalmetrics = read.csv(file.path(modelresultspath,STRESULTSPATH))

# changing model names
dataevalmetrics$model = factor(dataevalmetrics$model, levels = MODELS)
levels(dataevalmetrics$model) = MODELNAMES

dataevalmetrics[dataevalmetrics$element %in% 'Ca43','element'] = 'Ca'

## summary

msumm_st =  dataevalmetrics%>% group_by(element,model) %>% 
  summarise(r2 = mean(r2),
            rmse =  mean(rmse),
            prmse = mean(prmse),
            mae = mean(mae))%>% ungroup()

## element and model names order

idordered =  dataevalmetrics%>% 
  group_by(element) %>% summarise(r2 = mean(r2))%>%arrange(desc(r2)) %>%pull(element)

modelordered =  dataevalmetrics%>%
  group_by(model) %>% summarise(r2 = mean(r2))%>% arrange(r2) %>%pull(model)


## changing order
dataevalmetrics$element = factor(dataevalmetrics$element, 
                            levels = c(idordered))
dataevalmetrics$model = factor(dataevalmetrics$model, 
                               levels = c(modelordered))
msumm_st$model = factor(msumm_st$model, 
                        levels = c(modelordered))
## plots

plotsr2 = ggplot(data = dataevalmetrics%>%
                   filter(!r2<0), 
                 aes(colour =  factor(model, 
                                      levels = modelordered))) + 
  geom_point(aes(r2,element, 
                 group =factor(model, 
                               levels = modelordered)),
             alpha = 0.3, 
             size = 0.2,position = position_dodge(width = 0.8)) + 
  geom_point(data = msumm_st%>%filter(!r2<0) ,
             aes(r2,element,
                 group = factor(model, 
                                levels = modelordered), 
                 shape = factor(model, 
                                levels = modelordered)), 
             size = 1.7,position = position_dodge(width = 0.8))+#, shape=15) + 
  labs(y = 'Elements\n', x =  paste("R²"), color = "Regression \nmodels") + 
  theme_Publication(base_size = 18,legendsize = 14, legendposition = 'top',
                    base_family="helvetica" ,legendtitlesize = 16) + #scale_color_brewer(palette="Dark2")+
  facet_grid(factor(element, 
                    levels = c(idordered)) ~ ., scales = 'free_y')+
  theme(strip.background = element_blank(),strip.text.y = element_blank(),
  )+
  theme(panel.spacing = unit(0.6, "lines"))+
  scale_colour_manual(name = "Models",
                      labels = levels(msumm$model),
                      values = brewer.pal(5, "Dark2")) +   
  scale_shape_manual(name = "Models",
                     labels = levels(msumm$model),
                     values = c( 10, 15, 16,17,18))+
  scale_x_continuous(limits=c(0.2, 0.85), breaks=seq(0.2, 0.9, 0.2))+ 
  guides(color = FALSE, shape= FALSE)

###### Multi target results

mtdata = read.csv(file.path(modelresultspath,MTRESULTSPATH))

# changing model names
mtdata $model = factor(mtdata $model, levels = MODELS)
levels(mtdata $model) = MODELNAMES

mtdata = mtdata %>% 
  mutate( chain = str_replace(chain,pattern = 'Ca43', 'Ca'),
          element = str_replace(element,pattern = 'Ca43', 'Ca')) %>%
  mutate(descrlb = paste(element, chain, model)) %>% ungroup()

## summary

best_models_chain = mtdata%>%
  group_by(element, model,chain, descrlb)%>%
  summarise_all(mean)%>%
  group_modify(~.x%>% arrange(desc(r2))%>% 
                 slice(1L))%>%ungroup()

msumm_mt = mtdata%>% group_by(element,model) %>% 
  summarise(r2 = mean(r2),
            prmse = mean(prmse),
            rmse =  mean(rmse))%>% ungroup() 


## changing order
mtdata$element = factor(mtdata$element, 
                                 levels = idordered)
mtdata$model = factor(mtdata$model, 
                               levels = modelordered)

msumm_mt$model= factor(msumm_mt$model, 
                       levels = modelordered)


### plot multitarget
plotsr2mt = ggplot(data = mtdata%>%
                     filter(!r2<0), ### Cd svr linear regression results
                   aes(colour =  factor(model, 
                                        levels = modelordered))) + 
  geom_point(aes(r2,element, 
                 group =factor(model, 
                               levels = modelordered)),
             alpha = 0.3, 
             size = 0.2,position = position_dodge(width = 0.8)) + 
  geom_point(data = msummultitarget%>%filter(!r2<0) ,
             aes(r2,element,
                 group = factor(model, 
                                levels = modelordered), 
                 shape = factor(model, 
                                levels = modelordered)), 
             size = 1.7,position = position_dodge(width = 0.8))+#, shape=15) + 
  labs(y = '\n', x = paste("R²"), color = "Regression \nmodels") + 
  
  theme_Publication(base_size = 18,legendsize = 14, legendposition = 'top',
                    base_family="helvetica" ,legendtitlesize = 16) + #scale_color_brewer(palette="Dark2")+
  facet_grid(factor(element, 
                    levels = c(idordered)) ~ ., scales = 'free_y')+
  theme(strip.background = element_blank(),strip.text.y = element_blank(),
        #legend.title = element_text(face="italic", size = 14)
        panel.spacing = unit(0.6, "lines"),
        legend.title = element_text(face="bold",size = 16))+
  scale_colour_manual(name = "Learning models: ",
                      labels = levels(msumm_mt$model),
                      values = brewer.pal(5, "Dark2")) +   
  scale_shape_manual(name = "Learning models: ",
                     labels = levels(msumm_mt$model),
                     values = c( 10, 15, 16,17,18))+
  scale_x_continuous(limits=c(0.2, 0.85), breaks=seq(0.2, 0.9, 0.2))


### merging plots, figure 3


leg <- get_legend(plotsr2mt)

r2 = ggarrange(ggarrange(plotsr2,
                         plotsr2mt+ 
                           guides(color = FALSE, shape= FALSE),
                         labels = c("A","B"),
                         font.label = list(size = 16),
                         ncol = 2),
               NULL,
               leg,
               nrow = 3,
               heights = c(0.9,-0.03,.12))%>%ggexport( 
                 filename = file.path(PLOTS_OUTPUTPATH,"figure_3_summaryr2_st_mt.jpg"), 
                 width  = 11000,height = 10000, res = 1800)

####
### FIGURE 4: differences between best prediction models

### single target
bestsingletarget =  dataevalmetrics%>% group_by(element,model) %>% summarise(
  r2_st = mean(r2),
  rmse_st =  mean(rmse),
  prmse_st = mean(prmse),
  mae_st = mean(mae))%>%
  group_modify(~.x%>% arrange(desc(r2_st))%>% 
                 slice(1L))%>%ungroup()%>%
  rename(model_st = model)

##
bestsingletarget_cv = do.call(rbind,lapply(unique(dataevalmetrics$element),function(eoi){
  noisubset = dataevalmetrics%>%filter(element%in%eoi)
  bestref = bestsingletarget%>%
    filter(element%in%eoi)
  
  noisubset%>%
    filter(model%in%unique(bestref$model_st))
})) %>%rename(r2_st = r2,
              rmse_st = rmse,
              prmse_st = prmse,
              model_st = model)


### multi target


best_chainpernutrient = best_models_chain %>% group_by(element)%>%
  group_modify(~.x%>% arrange(desc(r2))%>% 
                 slice(1L))%>%ungroup()%>%
  rename(model_mt = model)

best_chainpernutrient = best_models_chain %>% group_by(element)%>%
  group_modify(~.x%>% arrange(r2)%>% 
                 slice(1L))%>%ungroup()%>%
  rename(model_mt = model)

##

best_models_chain_cv = do.call(rbind,lapply(unique(mtdata$element),function(eoi){
  noisubset = mtdata%>%filter(element%in%eoi)
  bestref = best_chainpernutrient%>%
    filter(element%in%eoi)
  
  noisubset%>%
    filter(model%in%unique(bestref$model_mt))
})) %>%rename(r2_mt = r2,
              rmse_mt = rmse,
              prmse_mt = prmse,
              model_mt = model)


###

bestresultsstmt_cv = best_models_chain_cv %>%
  left_join(bestsingletarget_cv%>%
              select(element, cv, model_st, r2_st, prmse_st,rmse_st ), by = c('element', 'cv'))


### estimating significance difference
eoi = 'Li'
signifr2 = do.call(rbind,lapply(as.character(unique(bestresultsstmt_cv$element)), function(eoi){
  
  singlecvresults = bestresultsstmt_cv%>%filter(as.character(element)%in%eoi)

  wilcox.test(singlecvresults$prmse_st,singlecvresults$prmse_mt,paired=TRUE)
  
  tests = t.test(singlecvresults$r2_st,singlecvresults$r2_mt)
  tests = wilcox.test(singlecvresults$r2_st,singlecvresults$r2_mt,paired=TRUE)
  #tests = wilcox.test(singlecvresults$prmse_st,singlecvresults$prmse_mt,paired=TRUE)
  
  labeldif = significance_diference_labels(tests$p.value,
                                           labels_sign_values = c("***"=0.001, "**"=0.01,  "*"=0.05,"ns" = 1))
  
  data.frame(element = eoi, pvalue = tests$p.value, label = labeldif)
  
}))

signifr2%>%
  filter(!label%in%"ns")%>%
  pull(element)

## merging label

bestresultsstmt_cv = bestresultsstmt_cv %>% left_join(signifr2, by='element')

bestresultsstmt_cvlongerr2 = bestresultsstmt_cv%>%select(c(element,chain, model_mt,model_st, 
                                                           descrlb,r2_st,r2_mt,rmse_st,rmse_mt,pvalue,label, cv))%>%
  pivot_longer(c(r2_st,r2_mt,rmse_st,rmse_mt))



##
summmarynoisigni = bestresultsstmt_cvlongerr2 %>% 
  group_by(element,name) %>% 
  summarise(
    model_mt = unique(model_mt),
    model_st = unique(model_st),
    mean_r2 = mean(value),
    sd_r2   = sd(value),
    label = unique(label),
    chain = unique(chain)
  ) 


summmarynoisigni%>%
  pivot_wider(id_cols = c(element,label),names_from = name, values_from = mean_r2)%>%
  left_join(summmarynoisigni %>%
              group_by(element)%>%
              summarise(
                chain = unique(chain),
                mtmodel = unique(model_mt),
                stmodel = unique(model_st)),
            by = 'element')%>%write.csv("table4.csv")

signiflabels = summmarynoisigni %>% group_by(element) %>%
  summarise(
    max_r2 = max(mean_r2 + sd_r2),
    sp   = mean(sd_r2),
    label = unique(label)
  ) 


### per model
stmtdiffpermodel$model
stmtdiffpermodel = mtdata%>%rename(r2_mt = r2,
                rmse_mt = rmse,
                prmse_mt = prmse,
                mae_mt = mae)%>%
  left_join(  dataevalmetrics
              %>%rename(r2_st = r2,
                        rmse_st = rmse,
                        prmse_st = prmse,
                        mae_st = mae), by = c('element','model', 'cv'))%>%
  mutate(r2improve = (r2_mt-r2_st)/r2_st*100,
         rmseimprove = ((rmse_mt - rmse_st)*-1*rmse_st^(-1)*100))

## this is a mistake in the code those values have only one element in the chain but because there wasnot improvement
stmtdiffpermodel = stmtdiffpermodel%>%
  filter(!(element == 'Fe' & model == 'SVM-Linear'))%>%
  filter(!(element == 'Co' & model == 'SVM-Linear'))

# summary
msum2= stmtdiffpermodel%>%
  filter(!(element == 'Fe' & model == 'SVM-Linear'))%>%
  filter(!(element == 'Co' & model == 'SVM-Linear'))%>%
  group_by(element,model)%>%
  summarise_all(mean)

# element order

idl = c()
for(i in length(idlabels):1){
  idl = c(idl,idlabels[i])
}

msum2$element = factor(msum2$element, levels = idlabels)
stmtdiffpermodel$element = factor(stmtdiffpermodel$element, levels = idlabels)
round(msum2%>%
  filter(element =='Fe'& model == 'RF')%>%
  select(element,model,r2improve,rmseimprove )%>%
  pull(r2improve),1)
round(msum2%>%
        filter(element =='Fe'& model == 'RF')%>%
        select(element,model,r2improve,rmseimprove )%>%
        pull(rmseimprove),1)

msum2%>%
        filter(element =='Mo')

### plots

removechain = c('Fe' ,'SVM-Linear')

plotmtr2 = ggplot(data = stmtdiffpermodel%>%filter(!r2improve< -15)%>%
                    filter(!r2improve> 15), 
                  aes(colour =  model)) + 
  geom_vline(xintercept = 0, color = 'black')+
  geom_point(aes(r2improve,element , shape=model
  ),alpha = 0.3, 
  size = 0.6,position = position_dodge(width = 0.5)) + 
  geom_point(data = msum2 ,
             aes(r2improve,element, 
                 group = model, 
                 shape = model), 
             size = 2,
             position = position_dodge(width = 0.5)) + 
  
  labs(y ="Elements\n", color = " ", x = paste("Δ R² (%)")) +
  
  theme_Publication(base_size = 18,legendsize = 14, legendposition = 'top',
                    base_family="helvetica" ,legendtitlesize = 16) +
  theme(panel.grid = element_blank(),
        panel.border = element_blank(),
        strip.background = element_blank(),
        #axis.ticks.y = element_blank(),
        #axis.text.y = element_blank(),
        strip.text.y = element_blank(),
        axis.title.y=element_text(angle=90),
        #legend.title = element_text(face="italic", size = 13)
  )+
  theme(panel.spacing = unit(0.6, "lines"),
        #plot.margin = unit(c(0.2,.2,0.2,.2), "cm")
  )+
  facet_grid(element ~ ., scales = 'free_y')+
  scale_colour_manual(name = "Models",
                      labels = levels(stmtdiffpermodel$model),
                      values = brewer.pal(5, "Dark2")) +   
  scale_shape_manual(name = "Models",
                     labels = levels(stmtdiffpermodel$model),
                     values = c( 10, 15, 16,17,18))+
  scale_x_continuous(limits=c(-10, 25), breaks=seq(-10, 25, 5))

## rmse

plotmtrmse = ggplot(data = stmtdiffpermodel%>%filter(!r2improve< -15)%>%
                      filter(!r2improve> 15), 
                    aes(colour =  model)) + 
  geom_vline(xintercept = 0, color = 'black')+
  geom_point(aes(rmseimprove,element , shape=model
  ),alpha = 0.3, 
  size = 0.6,position = position_dodge(width = 0.5)) + 
  geom_point(data = msum2 ,
             aes(rmseimprove,element, 
                 group = model, 
                 shape = model), 
             size = 2,
             position = position_dodge(width = 0.5)) + 
  
  labs(y ="\n", color = " ", x = paste("Δ RMSE(%)")) +
  
  theme_Publication(base_size = 18,legendsize = 14, legendposition = 'top',
                    base_family="helvetica" ,legendtitlesize = 16) + 
  theme(panel.grid = element_blank(),
        panel.border = element_blank(),
        strip.background = element_blank(),
        #axis.ticks.y = element_blank(),
        #axis.text.y = element_blank(),
        strip.text.y = element_blank(),
        axis.title.y=element_text(angle=90),
        panel.spacing = unit(0.6, "lines"),
        #plot.margin = unit(c(0.2,.2,0.2,.2), "cm")
        legend.title = element_text(face="bold",size = 16))+
  facet_grid(element ~ ., scales = 'free_y')+
  scale_colour_manual(name = "Learning models: ",
                      labels = levels(stmtdiffpermodel$model),
                      values = brewer.pal(5, "Dark2")) +   
  scale_shape_manual(name = "Learning models: ",
                     labels = levels(stmtdiffpermodel$model),
                     values = c( 10, 15, 16,17,18))+
  scale_x_continuous(limits=c(-10, 25), breaks=seq(-10, 25, 5))

### merging plots, figure 4


leg <- get_legend(plotmtrmse)

pl1 = ggarrange(ggarrange(plotmtr2+ 
                            guides(color = FALSE, shape= FALSE),
                          plotmtrmse+ 
                            guides(color = FALSE, shape= FALSE),
                          labels = c("A","B"),
                          font.label = list(size = 16),
                          ncol = 2) ,
                NULL,
                leg,
                labels = c("",""),
                nrow = 3,
                heights = c(0.9,-0.03,.12))
pl1%>%ggexport( 
  filename = file.path(PLOTS_OUTPUTPATH,"figure_4_differencesr2rmse.jpg"), 
  width  = 12000,height = 10000, res = 1800)



####
### FIGURE 5: predictions using test dataset

testresults = read.csv(file.path(modelresultspath,TESTRESULTSPATH))

elementsoi =  unique(summmarynoisigni%>%
  filter(!label%in% 'ns')%>%
  pull(element))

models = c("SVM-Linear","PLS", "Lasso", "Ridge","RF")


idlabels
pos = c()
for(eoi in elementsoi){
  for(i in 1:length(idlabels)){
    if(eoi == idlabels[i]){
      pos = c(pos, i)
    }
  }
}
names(pos)=elementsoi

elementsoi = names(sort(pos))

# changing model names

testresults = testresults%>%mutate(element = str_replace(element, 'Ca43', 'Ca'),
                                       chain = str_replace(chain, 'Ca43', 'Ca'))

i = 3
plotsm = lapply(1:length(elementsoi), function(i){


  
  noi = elementsoi[i]
  
  noipred = testresults %>% filter(element %in% noi)%>%
    mutate(regressortype=factor(regressortype, levels = c("multi_target",
                                                          "single_target")))
  noipred[noipred$regressortype%in%"single_target","chain"] = ''
  noipred[noipred$regressortype%in%"multi_target","chain"] = paste0(
    '',noipred[noipred$regressortype%in%"multi_target","chain"])
  
  
  rmseval = caret::RMSE(obs = noipred$real,pred = noipred$pred)
  
  pointpos = as.numeric(str_locate(as.character(rmseval), pattern = "[.]")[1])
  if(pointpos>=4){
    decposition = 0
  }else if(pointpos==3){
    decposition = 1
  }else{
    decposition = 2
  }
  
  evalmetrics = noipred %>%mutate(regressor = regressortype)%>%
    group_by(regressortype) %>% 
    summarise(r2metric = round(caret::R2(obs = real,pred = pred, formula = 'traditional'),2),
              rmsemetric = round(caret::RMSE(obs = real,pred = pred), decposition))
  
  cat(noi,round(evalmetrics%>%
                  summarise(deltar2 = (r2metric[1]-r2metric[2])/r2metric[2]*100,
                            deltarmse = (rmsemetric[1]-rmsemetric[2])/rmsemetric[2]*100,)%>%
                  pull(deltarmse),1),"\n")
  
  label = noipred %>%mutate(regressor = regressortype)%>%group_by(regressortype) %>% 
    summarise(metric = paste0(element,"\n",regressor, '\n', chain," \nR² = ", 
                              round(caret::R2(obs = real,pred = pred, formula = 'traditional'),2),
                              "\nRMSE = ", 
                              round(caret::RMSE(obs = real,pred = pred), decposition), 
                              ' (ppm)'))%>% 
    
    mutate(metric = str_replace(metric, 'multi_target', 'Multi-target'))%>% 
    mutate(metric = str_replace(metric, 'single_target', 'Single-target'))
  
  
  noipred= noipred%>%left_join(label, by ='regressortype')
  minreal = round(min(noipred$real),decposition)
  maxreal = round(max(noipred$real),decposition)
  minpred = round(min(noipred$pred),decposition)
  maxpred = round(max(noipred$pred),decposition)
  basemodel = unique(noipred$model)
  if(length(basemodel)>0){
    pos = c()
    for(i in 1:length(basemodel)){
      if(basemodel[i] == "svr_linear"){
        pos = c(pos,1)
      }else{
        tmppos = which(str_detect(tolower(models), basemodel[i]))
        pos = c(pos,tmppos)
      }
    }
  }
  if(length(pos) == 2){
    pos = pos[c(2,1)]
  }
  cat("***",basemodel,"***\n")
  
  basemodel = models[pos]
  m = ggplot(noipred, aes(pred, real)) + 
    geom_point(aes(color=model, shape=model), size = 2)+
    geom_smooth( method = "lm", color = "black") + facet_wrap(~metric) + 
    labs(title = "", y = '',
         x = "") + 
    theme_Publication(base_size = 22,base_family = "Helvetica")+
    theme(
      strip.text.x = element_text(size = 17),
      strip.background =element_rect(fill="white"),
      panel.spacing = unit(1, "lines"),###space between panelspanel.spacing = unit(2, "lines")
      plot.margin = margin(.01, .2,.01,.2, "cm"))+
    
    scale_y_continuous(breaks = seq(minreal, maxreal,round(abs(minreal - maxreal)/1.8, decposition)))+
    scale_x_continuous(breaks = seq(minpred, maxpred,round(abs(minpred - maxpred)/1.8, decposition)))+
    scale_colour_manual(name = "Models",
                        labels = models ,
                        values = brewer.pal(5, "Dark2")[pos])+   
    scale_shape_manual(name = "Models",
                       labels = models,
                       values = c( 10, 15, 16,17,18)[pos])+ 
    guides(color = FALSE, shape= FALSE)
  

  datar = noipred%>%mutate(regressor = regressortype)%>%
    mutate(regressortype=factor(regressortype, levels = c("multi_target",
                                                          "single_target")))%>%
    group_by(regressortype)%>%
    summarise(r2 = round(caret::R2(obs = real,pred = pred, formula = 'traditional'),2),
              rmse = round(caret::RMSE(obs = real,pred = pred),2))
  cat('\n')
  cat(noi,
      'r2: ', round(((datar$r2[1]-datar$r2[2])),2),
      'rmse: ', round(((datar$rmse[1]-datar$rmse[2])),decposition))
  cat('\n')
  cat(noi,
      'r2: ', round(((datar$r2[1]-datar$r2[2])/datar$r2[2])*100,1),
      'rmse: ', round(((datar$rmse[1]-datar$rmse[2])/datar$rmse[2])*100,1))
  cat('\n')

  
  m

})




### get legend

graph = ggplot(data = testresults,#%>%filter(!prmse>90), 
               aes(colour =  model,
                   group = model, 
                   shape = model)) + 
  
  geom_point(aes(real,pred),size = 4)+#, shape=15) + 
  
  scale_colour_manual(name = "Learning models: ",
                      labels = models[-2],
                      values = brewer.pal(5, "Dark2")[c(1,3:5)]) +   
  scale_shape_manual(name = "Learning models: ",
                     labels = models[-2],
                     values = c( 10, 16,17,18))+
  theme_Publication(base_size = 16,base_family = "Helvetica",
                    legendposition = "bottom", legendsize = 28,
                    legendtitlesize = 30)+
  theme(legend.title = element_text(face="bold",size = 30))

leg <- get_legend(graph)

pl1 = ggarrange(ggarrange(
  plotlist = plotsm[1:3],
  ncol = 3),
  NULL,
  ggarrange(
    plotlist = plotsm[4:6],
    ncol = 3),
  NULL,
  ggarrange(
    plotlist = plotsm[7:9],
    ncol = 3),
  NULL,
  ggarrange(
    plotsm[[10]],
    NULL,
    NULL,
    ncol = 3),
  #leg,
  nrow = 7,
  heights = c(1,-0.11,1,-0.11,1,-0.11,1)
)

pl1 = annotate_figure(pl1, 
                      left = text_grob("Observed element concentration (ppm)", 
                                       size = 28,rot = 90, vjust = 1, face = "bold"),# gp = gpar(cex = 1.3)),
                      bottom = text_grob("Predicted element concentration (ppm)", size = 28, 
                                         face = "bold"),# gp = gpar(cex = 1.3), size = 16),
)

ggarrange(pl1,
          NULL,
          leg,
          nrow = 3,
          heights = c(0.9,-0.05,0.15))%>%ggexport( 
            filename = file.path(PLOTS_OUTPUTPATH,"figure_5_testresults.jpg"),
            width  = 19000,height = 15500, res = 1000)

