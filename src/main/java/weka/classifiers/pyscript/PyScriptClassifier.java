package weka.classifiers.pyscript;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.BatchPredictor;
import weka.core.CapabilitiesHandler;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.python.PythonSession;

/**
 * WEKA class that calls an arbitrary Python script that can take
 * training and testing instances (in the form of Numpy arrays)
 * and return predictions.
 * @author cjb60
 *
 */
public class PyScriptClassifier extends AbstractClassifier implements
	  BatchPredictor, CapabilitiesHandler {
	
	private static final long serialVersionUID = 2846535265976949760L;
	
	/**
	 * Default values for the parameters.
	 */
	private final String DEFAULT_PYFILE = "scripts/test.py";
	private final String DEFAULT_PYFILE_PARAMS = "";
	
	private Instances m_trainingData = null;
	private boolean m_debug = false;
	
	private Filter m_nominalToBinary = null;
	private Filter m_replaceMissing = null;
	
	/** The default Python script to execute **/
	private String m_pyFile = DEFAULT_PYFILE;
	
	/** If there are any parameters to pass to the script **/
	private String m_pyFileParams = DEFAULT_PYFILE_PARAMS;
	
	public String getPythonFile() {
		return m_pyFile;
	}
	
	public void setPythonFile(String pyFile) {
		m_pyFile = pyFile;
	}
	
	public String getPythonFileParams() {
		return m_pyFileParams;
	}
	
	public void setPythonFileParams(String pyFileParams) {
		m_pyFileParams = pyFileParams;
	}

	@Override
	public void buildClassifier(Instances data) throws Exception {
		
		/*
		 * Prepare training data for script
		 */
		
		m_replaceMissing = new ReplaceMissingValues();
		m_replaceMissing.setInputFormat(data);
		data = Filter.useFilter(data, m_replaceMissing);
		
	    m_nominalToBinary = new NominalToBinary();
	    m_nominalToBinary.setInputFormat(data);
	    m_trainingData = Filter.useFilter(data, m_nominalToBinary);
	}

	@Override
	public void setBatchSize(String size) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public String getBatchSize() {
		// TODO Auto-generated method stub
		return null;
	}
	
	private String loadFile(String filename) throws IOException {
		List<String> contents = Files.readAllLines(Paths.get(filename));
		StringBuffer sb = new StringBuffer();
		for(String line : contents) {
			sb.append(line + "\n");
		}
		return sb.toString();
	}

	@Override
	@SuppressWarnings("unchecked")
	public double[][] distributionsForInstances(Instances insts)
			throws Exception {
		
	    if (!PythonSession.pythonAvailable()) {
	        // try initializing
	        if (!PythonSession.initSession("python", m_debug)) {
	          String envEvalResults = PythonSession.getPythonEnvCheckResults();
	          throw new Exception("Was unable to start python environment: "
	            + envEvalResults);
	        }
	    }
	    
	    try {  
	    	
	    	insts = Filter.useFilter(insts, m_replaceMissing);
	        insts = Filter.useFilter(insts, m_nominalToBinary);
	        
	        int numClasses = insts.numClasses();
	        
	        // remove the class attribute
	        Remove r = new Remove();
	        r.setAttributeIndices("" + (insts.classIndex() + 1));
	        r.setInputFormat(insts);
	        insts = Filter.useFilter(insts, r);
	        insts.setClassIndex(-1);
	    	
		    PythonSession session = PythonSession.acquireSession(this);
		    
		    /*
		     * Ok, push the training data to Python. The variables will be called
		     * X and Y, so let's execute to script to rename these.
		     */
		    session.instancesToPythonAsScikietLearn(m_trainingData, "train", m_debug);
		    session.executeScript("X_train = X\ny_train=Y", m_debug);
		    
		    /*
		     * Push the test data. These will also be X and Y, so have a
		     * script that renames these to X_test and y_test.
		     */
		    session.instancesToPythonAsScikietLearn(insts, "test", m_debug);
		    session.executeScript("X_test = X\n", m_debug);
		    
		    System.out.format("train, test = %s, %s\n",
		    		m_trainingData.numInstances(), insts.numInstances());
		    
		    /*
		     * Tell the script that it is being invoked by WEKA and pass
		     * some params to it.
		     */
		    session.executeScript("use_weka = True\n", m_debug);
		    //session.executeScript("weka_params = " + getPythonFileParams() + "\n", m_debug);
		    session.executeScript("preds = None\n", m_debug);
		    
		    /*
		     * Ok, now this script should recognise X_train, y_train,
		     * X_test, and y_test.
		     */
		    String pyFile = loadFile( getPythonFile() );
		    List<String> outAndErr = session.executeScript(pyFile, true);
		    System.out.println(outAndErr.get(0));
		    
		    double[][] distributions = new double[insts.numInstances()][numClasses];
		    
			List<Object> preds = 
		    	(List<Object>) session.getVariableValueFromPythonAsJson("preds", m_debug);
		    for(int y = 0; y < insts.numInstances(); y++) {
		    	Object vector = preds.get(y);
		    	double[] probs = new double[numClasses];
				List<Double> probsForThisInstance = (List<Double>) vector;
		    	for(int x = 0; x < probs.length; x++) {
		    		probs[x] = probsForThisInstance.get(x);
		    	}
		    	distributions[y] = probs;
		    }
			
			return distributions;
			
	    } catch(Exception ex) {
			ex.printStackTrace();
		} finally {
			PythonSession.releaseSession(this);
		}
	    
	    return null;
	}
	
	@Override
	public void setOptions(String[] options) throws Exception {
		String tmp = Utils.getOption("F", options);
		if(tmp.length() != 0) { 
			setPythonFile(tmp);
		}
		tmp = Utils.getOption("P", options);
		if(tmp.length() != 0) {
			setPythonFileParams(tmp);
		}
	}
	
	@Override
	public String[] getOptions() {
		Vector<String> result = new Vector<String>();
		result.add("-F");
		result.add( "" + getPythonFile() );
		result.add("-P");
		result.add( "" + getPythonFileParams() );
		Collections.addAll(result, super.getOptions());
	    return result.toArray(new String[result.size()]);
	}
	
	public static void main(String[] argv) {
		runClassifier(new PyScriptClassifier(), argv);
	}

}
