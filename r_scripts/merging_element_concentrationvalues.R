#######
#####     Concatening the files into a single file
#######
rm(list = ls())
library(tidyverse)

read_individualfile = function(fn,path, pattern = '.csv', add_id = TRUE){
  
  data = read_csv(file.path(path,fn))%>%
    mutate(element = str_sub(fn,1,str_locate(fn, pattern = pattern)[1]-1))
  
  data%>%
    mutate(id = apply(data[,c(1,c(50,55),c(300:305),c(390,394))], 1, function(x) paste0(round(x,4), collapse = '_')))
  
}

### CONFIGURATION

inputfolder = "data/"

### fining unique id

elements = list.files(inputfolder, pattern = '*.csv')
idelementdata = do.call(rbind, lapply(1:length(elements),function(x) 
  read_individualfile(elements[x], inputfolder)%>%
                         select(id, element)))

uniqueids = unique(idelementdata$id)

### finding which elements share the same data

dataid = data.frame(id = uniqueids, element = 'Test')
x = 1
for(x in 1:length(elements)){
  dataid = dataid%>%
    left_join(read_individualfile(elements[x], inputfolder)%>%
                select(id, element), by = 'id')
}

dataid = dataid%>%na.omit()%>%pull(id)


### create an unique file for the elements and the spectral data

x = 1
mergeddata = do.call(rbind,lapply(1:length(elements), function(x){
  read_individualfile(elements[x], inputfolder)%>%
                filter(id %in% dataid)%>%
    pivot_longer(starts_with('WL'))%>%
    select(id, date,element, str_sub(elements[x],1,str_locate(elements[x], pattern = '.csv')[1]-1),name, value)%>%
    rename(concentration = str_sub(elements[x],1,str_locate(elements[x], pattern = '.csv')[1]-1))
}))


elementconcentrationdata = mergeddata %>%
  select(id, element, concentration)%>%
  pivot_wider(id, names_from = element, values_from = concentration, values_fn = unique)
  

spectraldata = mergeddata %>%
  select(id, name, value)%>%
  pivot_wider(id, names_from = name, values_from = value, values_fn = unique)

# there are 742 observation that share similar spectral data
data = elementconcentrationdata%>%
  left_join(spectraldata, by = 'id')%>%
  na.omit()

data
### export

data%>%
  select(B:Zn)%>%
  write.csv("data/grouped/nutrient_values_.csv")


data%>%
  select(starts_with('WL.'))%>%
  write.csv("data/grouped/spectral_data_.csv")

