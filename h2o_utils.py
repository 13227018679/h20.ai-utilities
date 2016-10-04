import unicodedata
import h2o
from os import path, remove
import sys
import StringIO
import csv

SILENCE = False

METRIC_NAMES = ["AUC_tr", "AUC_val", "AUC_x", "AUC_test"]
def print_model_evaluation(model, test_df):
    if test_df is not None:
        test_auc = model.model_performance(test_df).auc()
    else:
        test_auc = 0
    
    metrics = [
        model.auc(train=True),
        model.auc(valid=True) if model.auc(valid=True) is not None else 0 ,
        model.auc(xval=True) if model.auc(xval=True) is not None else 0 ,
        test_auc
    ]

    print "\t\t\t" + "\t".join(METRIC_NAMES)
    print model.model_id,
    print "\t" + "\t".join(map(lambda m: str(round(m,3)), metrics))

    print "\n\nVariable Name\t\t Relative Importance"
    def format_varimp_line(name, imp):
        return name + " "*(30-len(name)) + " " + str(round(imp,2))
    print "\n".join(map(lambda tup: format_varimp_line(tup[0], tup[2]), model.varimp()[:15]))

def silent_call(func, kwargs={}):
    if SILENCE:
        stdout = sys.stdout
        sys.stdout = StringIO.StringIO()
        try:
            result = func(**kwargs)
        finally:
            sys.stdout = stdout
        return result

    else:
        return func(**kwargs)


def import_csv_as_df(csv_path, dest_frame):
    print "Importing %s" % csv_path
    print "Converting to ASCII..."
    assert path.exists(csv_path), "Could not find: %s" % csv_path
    temp_ascii_path = csv_path + "_as_ascii_temp"
    with open(csv_path) as unicode_file:
        with open(temp_ascii_path, 'w') as ascii_file:
            for chunk in iter(lambda: unicode_file.read(16*4096), b''):
                unicode_data = chunk.decode('UTF-8')
                ascii_file.write(unicodedata.normalize('NFKD', unicode_data).encode('ASCII', 'ignore'))

    print "Loading into H2O..."
    data_df = silent_call(h2o.import_file, {"path": temp_ascii_path, "destination_frame": dest_frame})

    # remove(temp_ascii_path)

    data_df = force_column_types(data_df, csv_path.replace("model_data", "column_types"))

    return data_df

ANATELLA_H2O_TYPE_MATCH = {"U": "enum", "F": "int", "K": "int"}    
def force_column_types(data_df, column_types_csv):
    with open(column_types_csv, "r") as f:
        for row in csv.reader(f):
            col_name, col_type = row
            col_type = ANATELLA_H2O_TYPE_MATCH[col_type]
            if data_df.types[col_name] == col_type or col_name.endswith("CATEGORY"):
                convert_column_type(data_df, col_name, col_type)
    return data_df

def convert_column_type(data_df, col_name, to_type):
    try:
        if to_type=="int":
            data_df[col_name] = data_df[col_name].asnumeric()
        elif to_type=="enum":
            data_df[col_name] = data_df[col_name].asfactor()
            nb_levels = len(data_df[col_name].levels()[0])
            if nb_levels<=1:
                print "Warning: column '%s' has only '%i' level!" % (col_name, nb_levels)
            elif nb_levels>500:
                print "Warning: column '%s' has '%i' levels; too many!" % (col_name, nb_levels)
        else:
            raise AssertionError("Unhandled column-type '%s' for column '%s'!" % (col_type, col_name))
    except Exception as e:
        raise AssertionError("An error occured while converting column '%s' to '%s'!\n%s" % (col_type, col_name, str(e)))
            



def train_model(model, target_variable, train_df, val_df):
    print "Training model..." 
    kwargs = {"y": target_variable, 
        "x": range(0, train_df.ncol-1), 
        "training_frame": train_df
        }
    if val_df is not None:
        kwargs["validation_frame"] = val_df
     
    silent_call(model.train, kwargs)


USE_VAL_AND_TEST_FRAMES = False
def get_trained_model(data_df, target_variable, suffix):
    n_rows = data_df.nrow
    assert n_rows > 10000, "Only %i rows present to train model '%s'! This is too little!" % (n_rows, suffix) 
    n_targets = data_df[target_variable].asnumeric().sum()
    assert n_targets > 1000, "Only %i targets present to train model '%s'! This is too little!" % (n_targets, suffix) 
    print "Dataset with %i rows and %i targets (%s%%)" % (n_rows, n_targets, round(100.0*n_targets/n_rows,2)) 

    if USE_VAL_AND_TEST_FRAMES:
        train_df, val_df, test_df = data_df.split_frame([0.7, 0.2], 
            destination_frames = ["train_%s" % suffix, "val_%s" % suffix, "test_%s" % suffix])
    else:
        train_df = data_df
        val_df = None
        test_df = None

    assert len(data_df[target_variable].levels()[0])==2, \
        "The target-variable '%s' has %i levels bit it must be binary!\n" % (target_variable, len(data_df[target_variable].levels()[0])) + \
        "Instead found: %s" % data_df[target_variable].levels()[0]
    

    # from h2o.estimators.glm import H2OGeneralizedLinearEstimator
    # model = H2OGeneralizedLinearEstimator(model_id="BinomialGLM_%s" % suffix, nfolds=5, family = "binomial") # 
    # train_model(model, target_variable, train_df, val_df)
    # print_model_evaluation(model, test_df)

    from h2o.estimators.gbm import H2OGradientBoostingEstimator
    model = H2OGradientBoostingEstimator(model_id="BernoulliGBM_%s" % suffix, nfolds=5)
    train_model(model, target_variable, train_df, val_df)
    print_model_evaluation(model, test_df)

    # from h2o.estimators.random_forest import H2ORandomForestEstimator
    # model = H2ORandomForestEstimator(model_id="RandomForest_%s" % suffix, nfolds=5)
    # train_model(model, target_variable, train_df, val_df)
    # print_model_evaluation(model, test_df)

    return model