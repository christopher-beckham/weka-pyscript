package weka.classifiers.pyscript;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.rules.ZeroR;
import weka.core.Attribute;
import weka.core.BatchPredictor;
import weka.core.Capabilities;
import weka.core.CapabilitiesHandler;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.OptionMetadata;
import weka.core.Randomizable;
import weka.core.TechnicalInformation;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WekaException;
import weka.core.TechnicalInformation.Type;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.Standardize;
import weka.pyscript.Utility;
import weka.python.PythonSession;

/**
 * WEKA class that calls an arbitrary Python script that can take
 * training and testing instances (in the form of Numpy arrays)
 * and return predictions.
 * @author cjb60
 */
public class PyScriptClassifier extends AbstractClassifier implements BatchPredictor,
	TechnicalInformationHandler, CapabilitiesHandler {
	
	private static final long serialVersionUID = 2846535265976949760L;
	
	private final File DEFAULT_PYFILE = new File( System.getProperty("user.dir") );
	private final String DEFAULT_TRAIN_PYFILE_PARAMS = "";
	private final boolean DEFAULT_SHOULD_STANDARDIZE = false;
	private final boolean DEFAULT_SHOULD_BINARIZE = false;
	private final boolean DEFAULT_SHOULD_IMPUTE = false;
	private final String DEFAULT_PYTHON_COMMAND = "python";
	private final boolean DEFAULT_PRINT_STDOUT = false;
	private final boolean DEFAULT_SAVE_SCRIPT = false;
	
	private boolean m_shouldStandardize = DEFAULT_SHOULD_STANDARDIZE;
	private boolean m_shouldBinarize = DEFAULT_SHOULD_BINARIZE;
	private boolean m_shouldImpute = DEFAULT_SHOULD_IMPUTE;
	private String m_pythonCommand = DEFAULT_PYTHON_COMMAND;
	private String m_argsScript = null;
	protected String m_batchPredictSize = "100";
	private String m_pickledModel = null;
	private boolean m_printStdOut = DEFAULT_PRINT_STDOUT;
	
	private transient PythonSession m_session = null;
	
	private String m_modelString = null;
	
	private ZeroR m_zeror = null;
	
	/** The default Python script to execute */
	private File m_pyTrainFile = DEFAULT_PYFILE;
	
	/** If there are any parameters to pass to the training script */
	private String m_customArgs = DEFAULT_TRAIN_PYFILE_PARAMS;
	
	/** The Python script as a string (not null if m_saveScript is set) */
	private String m_pyScript = null;
	
	/** Save the script or not? */
	private boolean m_saveScript = DEFAULT_SAVE_SCRIPT;
	
	private Filter m_impute = null;
	private Filter m_standardize = null;
	private Filter m_binarize = null;
	
	public String getPickledModel() {
		return m_pickledModel;
	}
	
	@OptionMetadata(
		displayName = "saveScript", commandLineParamName = "save",
		description = "Save script in model?",
		commandLineParamSynopsis = "-save", commandLineParamIsFlag = true, displayOrder = 4
	)
	public boolean getSaveScript() {
		return m_saveScript;
	}
	
	public void setSaveScript(boolean b) {
		m_saveScript = b;
	}
	
	@OptionMetadata(
	   displayName = "printStdOut", commandLineParamName = "stdout",
	   description = "Print standard out from Python script to stderr?",
	   commandLineParamSynopsis = "-stdout", commandLineParamIsFlag = true, displayOrder = 4
	)
	public boolean getPrintStdOut() {
		return m_printStdOut;
	}
	
	public void setPrintStdOut(boolean b) {
		m_printStdOut = b;
	}
	
	@OptionMetadata(
		displayName = "pythonCommand",
		description = "Python executable command", commandLineParamName = "cmd",
		commandLineParamSynopsis = "-cmd <python executable>", displayOrder = 4
	)
	public String getPythonCommand() {
		return m_pythonCommand;
	}
	
	public void setPythonCommand(String s) {
		m_pythonCommand = s;
	}
	
	@OptionMetadata(
		displayName = "pythonFile",
		description = "Path to Python script", commandLineParamName = "script",
		commandLineParamSynopsis = "-script <path to script>", displayOrder = 4
	)
	public File getPythonFile() {
		return m_pyTrainFile;
	}
	
	public void setPythonFile(File pyFile) {
		m_pyTrainFile = pyFile;
	}
	
	@OptionMetadata(
		displayName = "arguments",
		description = "Arguments to pass to the script", commandLineParamName = "args",
		commandLineParamSynopsis = "-args <arguments>", displayOrder = 4
	)	
	public String getArguments() {
		return m_customArgs;
	}
	
	public void setArguments(String pyTrainFileParams) {
		m_customArgs = pyTrainFileParams;
	}
	
	@OptionMetadata(
		displayName = "shouldStandardize",
		description = "Should the data be standardized?", commandLineParamName = "standardize",
		commandLineParamSynopsis = "-standardize", commandLineParamIsFlag = true, displayOrder = 4
	)
	public boolean getShouldStandardize() {
		return m_shouldStandardize;
	}
	
	public void setShouldStandardize(boolean b) {
		m_shouldStandardize = b;
	}
	
	@OptionMetadata(
		displayName = "shouldBinarize",
		description = "Should nominal attributes be binarized?", commandLineParamName = "binarize",
		commandLineParamSynopsis = "-binarize", commandLineParamIsFlag = true, displayOrder = 4
	)
	public boolean getShouldBinarize() {
		return m_shouldBinarize;
	}
	
	public void setShouldBinarize(boolean b) {
		m_shouldBinarize = b;
	}
	
	@OptionMetadata(
		displayName = "shouldImpute",
		description = "Should missing values be imputed (with mean imputation)?", commandLineParamName = "impute",
		commandLineParamSynopsis = "-impute", commandLineParamIsFlag = true, displayOrder = 4
	)	
	public boolean getShouldImpute() {
		return m_shouldImpute;
	}
	
	public void setShouldImpute(boolean b) {
		m_shouldImpute = b;
	}
	
	@OptionMetadata(
		displayName = "batchSize",
		description = "How many instances should be passed to the model at testing time", commandLineParamName = "batch",
		commandLineParamSynopsis = "-batch <batch size>", displayOrder = 4
	)		
	@Override
	public String getBatchSize() {
		return m_batchPredictSize;
	}	
	
	@Override
	public void setBatchSize(String size) {
		m_batchPredictSize = size;
	}
	
	public String getModelString() {
		return m_modelString;
	}
	
	private void executeScript(String driver, String stdErrMessage) throws Exception {
		List<String> out = m_session.executeScript(driver, getDebug());
		if( stdErrMessage != null) {
			if(out.get(1).contains(Utility.TRACEBACK_MSG)) {
				throw new Exception(stdErrMessage + "\n" + out.get(1));
			}
		}
		if( getPrintStdOut() && !out.get(0).equals("") ) {
			System.err.println( "Standard out:\n" + out.get(0) );
		}
	}
	
	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();
		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.STRING_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);
		// class
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.NUMERIC_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);
		return result;
	}

	@Override
	public void buildClassifier(Instances data) throws Exception {
		
		getCapabilities().testWithFail(data);
		
		if(data.numAttributes() == 1 && data.classIndex() == 0) {
			m_zeror = new ZeroR();
			m_zeror.buildClassifier(data);
			return;
		}
		
		try {
		
			// see if the python file exists
			if( !getPythonFile().exists() ) {
				throw new FileNotFoundException( getPythonFile() + " doesn't exist!");
			}
			
			m_session = Utility.initPythonSession( this, getPythonCommand(), getDebug() );
			
			// set the working directory of the python vm to that of the script
			String parentDir = getPythonFile().getAbsoluteFile().getParent();
			String scriptName = getPythonFile().getName();
			if(parentDir != null) {
				String driver = "import os\nos.chdir(r'" + parentDir + "')\n";
				driver += "import sys\nsys.path.append(r'" + parentDir + "')\n";
				executeScript(driver, "An error happened while trying to change the working directory:");
			}
			
	    	// now load training and testing class
	    	String driver = "import imp\n"
	    			+ "cls = imp.load_source('cls','" + scriptName + "')\n";
	    	executeScript(driver, "An error happened while trying to load the Python script:");
	    	
		    if( getShouldImpute() ) {
		    	m_impute = new ReplaceMissingValues();
		    	m_impute.setInputFormat(data);
				data = Filter.useFilter(data, m_impute);
		    }
			if( getShouldStandardize() ) {
				m_standardize = new Standardize();
				m_standardize.setInputFormat(data);
				data = Filter.useFilter(data, m_standardize);
			}
			if( getShouldBinarize() ) {
				m_binarize = new NominalToBinary();
				m_binarize.setInputFormat(data);
		    	// make resulting binary attrs nominal, not numeric
				m_binarize.setOptions(new String[] { "-N" } );
		    	data = Filter.useFilter(data, m_binarize);
			}
			
			m_argsScript = Utility.createArgsScript(data, getArguments(), m_session, getDebug());		
			executeScript(m_argsScript, "An error happened while trying to create the args variable:");
		    
		    /*
		     * Ok, push the training data to Python. The variables will be called
		     * X and Y, so let's execute to script to rename these.
		     */
		    m_session.instancesToPythonAsScikitLearn(data, "train", false);
		    m_session.executeScript("args['X_train'] = X\nargs['y_train']=Y\n", getDebug());
		    
		    // build the classifier
		    driver = "model = cls.train(args)";	    
		    executeScript(driver, "An error happened while executing the train() function:");
		    
		    // save model parameters
		    m_pickledModel = m_session.getVariableValueFromPythonAsPickledObject("model", getDebug());
		    
		    // get model description
		    driver = "model_description = cls.describe(args, model)";
		    executeScript(driver, "An error happened while executing the describe() function:");
		    
		    m_modelString = m_session.getVariableValueFromPythonAsPlainString("model_description", getDebug());
		    
		    // save the script if needed
		    if( getSaveScript() ) {
		    	m_pyScript = String.join("\n", Files.readAllLines( getPythonFile().toPath() ) );
		    }
	    
		} catch(Exception ex) {
			ex.printStackTrace();
			throw ex;
		} finally {
			Utility.closePythonSession(this);
		}
	}
	
	@Override
	public double[] distributionForInstance(Instance inst)
			throws Exception {	
		Instances insts = new Instances(inst.dataset(), 0);
	    insts.add(inst);		
		return distributionsForInstances(insts)[0];		
	}	

	@Override
	public double[][] distributionsForInstances(Instances insts)
			throws Exception {
		
		if(m_zeror != null) {
			return m_zeror.distributionsForInstances(insts);
		}
		
	    try {
		
			//System.out.format("test = %s\n", insts.numInstances());
			//System.out.format("batch size = %s\n", getBatchSize());
			
			double[][] dists = new double[insts.numInstances()][insts.numClasses()];
			
		    m_session = Utility.initPythonSession(this, getPythonCommand(), getDebug());
	    	
			// see if the python file exists
			if( !getSaveScript() && !getPythonFile().exists() ) {
				throw new FileNotFoundException( getPythonFile() + " doesn't exist!");
			}    	

			// if the user didn't save the script
			String parentDir = null;
			String scriptName = null;
			if( !getSaveScript() ) {
				parentDir = getPythonFile().getAbsoluteFile().getParent();
				scriptName = getPythonFile().getName();
			} else {
				File tmp = Utility.tempFileFromString(m_pyScript);
				parentDir = tmp.getAbsoluteFile().getParent();
				scriptName = tmp.getName();
				if(getDebug()) System.err.println( "tmp python script: " + tmp.getAbsolutePath() );
			}
			
			if(parentDir != null) {
				String driver = "import os\nos.chdir(r'" + parentDir + "')\n";
				driver += "import sys\nsys.path.append(r'" + parentDir + "')\n";
				executeScript(driver, "An error happened while trying to change the working directory:");
			}
			
		    if( m_impute != null ) {
				insts = Filter.useFilter(insts, m_impute);
		    }
			if( m_standardize != null ) {
				insts = Filter.useFilter(insts, m_standardize);
			}
			if( m_binarize != null ) {
				m_binarize.setOptions(new String[] { "-N" } );
		    	insts = Filter.useFilter(insts, m_binarize);
			}
	        
	        int numClasses = insts.numClasses();
	        
	        // remove the class attribute
	        //Remove r = new Remove();
	        //r.setAttributeIndices("" + (insts.classIndex() + 1));
	        //r.setInputFormat(insts);
	        //insts = Filter.useFilter(insts, r);
	        //insts.setClassIndex(-1);
		    
	    	String driver = "import imp\n"
	    			+ "cls = imp.load_source('cls','" + scriptName + "')\n";
	    	executeScript(driver, "An error happened while trying to load the Python script:");
	    	executeScript(m_argsScript, "An error happened while trying to create the args variable:" );
		    
		    /*
		     * Push the test data. These will also be X and Y, so have a
		     * script that renames these to X_test and y_test.
		     */
		    m_session.instancesToPythonAsScikitLearn(insts, "test", false);
		    m_session.executeScript("args['X_test'] = X\n", getDebug());
		    
		    /*
		     * Push the weights of the saved model over.
		     */
		    m_session.setPythonPickledVariableValue("best_weights", m_pickledModel, getDebug());	     
		    
		    driver = "preds = cls.test(args, best_weights)";
		    executeScript(driver, "An error happened while executing the test() function:");
		    
			List<Object> preds = 
		    	(List<Object>) m_session.getVariableValueFromPythonAsJson("preds", getDebug());

			for(int y = 0; y < preds.size(); y++) {
		    	Object vector = preds.get(y);
		    	double[] probs = new double[numClasses];
				List<Double> probsForThisInstance = (List<Double>) vector;
		    	for(int x = 0; x < probs.length; x++) {
		    		probs[x] = probsForThisInstance.get(x);
		    	}
		    	dists[y] = probs;
			}
	    	
	    	return dists;
			
	    } catch(Exception ex) {
			ex.printStackTrace();
			throw ex;
		} finally {
			Utility.closePythonSession(this);
		}
	}
	
	@Override
	public String toString() {
		return m_modelString;
	}
	
	public String globalInfo() {  
		return "Class for calling classifiers that are Python scripts."
		+ "For more information, see\n\n"
		+ getTechnicalInformation().toString();
	}
	
	@Override
	public TechnicalInformation getTechnicalInformation() {
		return Utility.getTechnicalInformation();
	}
	
	@Override
	public boolean implementsMoreEfficientBatchPrediction() {
		return true;
	}
	
	public static void main(String[] argv) {
		runClassifier(new PyScriptClassifier(), argv);
	}

}
