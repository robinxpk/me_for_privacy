# Following data obtained from paper's github: https://github.com/chiragjp/voe
load("../data/nhanes9904_VoE.Rdata")

# Following copy-pasta from: voe/vibration/vibration_start.R
dat <- mainTab[, c('MORTSTAT', 'SDDSRVYR', 'WTMEC4YR', 'PERMTH_EXM', 'SES_LEVEL', 'RIDAGEYR', 'male', 'area', 'LBXT4', 'current_past_smoking', 'any_cad', 'any_family_cad', 'any_cancer_self_report', 'bmi', 'any_ht', 'any_diabetes', 'education', 'RIDRETH1', 'physical_activity', 'drink_five_per_day', 'LBXTC' )]

## base model is Cadmium (LBXBCD), age, sex; cluster(area) is to account for the correlated observations in NHANES
basemodel <- Surv(PERMTH_EXM, MORTSTAT) ~ scale(I(log(LBXT4+.1))) + RIDAGEYR + male + cluster(area) 

## compute vibration for LBXBCD, serum levels of the heavy metal cadmium, in association with time to death
dat <- subset(dat, !is.na(WTMEC4YR))
dat <- dat[complete.cases(dat), ]

# Prepare data for python Berkson-Code:
# Python-Code does not alter factor / string variables --> make sure all factor variables are actually factors / strings 
# Keep only truly numerical values as numerics
str(dat)
factor_vars = c(
    # -- Survival Indicator
    # "MORTSTAT",
    # -- Data set indicator
    "SDDSRVYR",              
    # -- Exam sample weight (combined)
    # "WTMEC4YR",
    # -- Follow-up time in month
    # i.e. this is "time" variable
    # "PERMTH_EXM",            
    # -- socioecnomic status level
    "SES_LEVEL",
    # -- Age at screening
    # "RIDAGEYR",              
    # -- Is sex male
    "male",
    # -- Area of where subject lives
    "area",                  
    # -- Serum total thyroxine (mycro gram / dL)
    # "LBXT4",
    # -- Factor if current or past smoking
    "current_past_smoking",  
    # -- Indicator for coronary artery disease
    "any_cad",
    # -- Indicator that any family member has had heart attack or angina
    "any_family_cad",        
    # -- Indicator if any cancer self report
    "any_cancer_self_report",
    # -- BMI of subject
    # "bmi",                   
    # -- Indicator if any hyptertension
    "any_ht",
    # -- Indicator if patient has diabetes
    "any_diabetes",          
    # -- Categorical level of education
    "education",
    # -- ethnicity of subject
    "RIDRETH1",              
    # -- Ranking of physical activity
    "physical_activity",
    # -- Indicator for heavy drinking
    "drink_five_per_day"
    # -- Serum total cholesterol mg/dL
    # "LBXTC"         
)
# dat[, factor_vars] = as.character(dat[, factor_vars])

write.table(
    dat, 
    "../data/voe_data.csv", 
    sep = ";", 
    row.names = FALSE, 
    dec = "."
)


