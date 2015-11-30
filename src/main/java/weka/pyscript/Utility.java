package weka.pyscript;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.Standardize;
import weka.python.PythonSession;

/**
 * Helper functions for PyScript
 * @author cjb60
 *
 */
public class Utility {
	
	public static final String TRACEBACK_MSG = "Traceback (most recent call last):";

	
	/**
	 * Start up a Python session
	 * @param the requesting object
	 * @param pythonCommand the Python command
	 * @param debug print debug information?
	 * @return a Python session
	 * @throws Exception if unable to start Python environment
	 */
	public static PythonSession initPythonSession(Object requester, String pythonCommand, boolean debug)
			throws Exception {
	    if (!PythonSession.pythonAvailable()) {
	        // try initializing
	        if (!PythonSession.initSession( pythonCommand, debug)) {
	          String envEvalResults = PythonSession.getPythonEnvCheckResults();
	          throw new Exception("Was unable to start python environment: "
	            + envEvalResults);
	        }
	    }
		PythonSession session = PythonSession.acquireSession(requester);  		
		return session;
	}
	
	/**
	 * Release Python session
	 * @param requester the requesting object
	 */
	public static void closePythonSession(Object requester) {
		PythonSession.releaseSession(requester);
	}
	
	/**
	 * 
	 * @param data the data to transform
	 * @param shouldImpute impute the data?
	 * @param shouldStandardize standardize the numeric attributes?
	 * @param shouldBinarize binarize the attributes?
	 * @return the transformed data
	 * @throws Exception
	 */
	public static Instances preProcessData(Instances data, boolean shouldImpute, 
			boolean shouldStandardize, boolean shouldBinarize) throws Exception {
	    if( shouldImpute ) {
	    	Filter impute = new ReplaceMissingValues();
	    	impute.setInputFormat(data);
			data = Filter.useFilter(data, impute);
	    }
		if( shouldStandardize ) {
			Filter standardize = new Standardize();
			standardize.setInputFormat(data);
			data = Filter.useFilter(data, standardize);
		}
		if( shouldBinarize ) {
			Filter binarize = new NominalToBinary();
			binarize.setInputFormat(data);
	    	// make resulting binary attrs nominal, not numeric
			binarize.setOptions(new String[] { "-N" } );
	    	data = Filter.useFilter(data, binarize);
		}
		return data;
	}
	
	private static String escape(String key) {
		return key.replace("'", "\\'").replace("\n", "\\n");
	}
	
	/**
	 * Create a script that pushes args to the Python VM
	 * @param df data frame
	 * @param session the Python session to send args to
	 * @param debug print debug information?
	 * @return the script to execute
	 * @throws Exception if an error traceback has been detected in stderr
	 */
	public static String createArgsScript(Instances df, String customParams,
			PythonSession session, boolean debug) throws Exception {
		
		StringBuilder script = new StringBuilder();
		
		script.append("args = dict()\n");
		
		// pass general information related to the training data
		if(df.classIndex() != -1) {
			script.append("args['num_classes'] = " + df.numClasses() + "\n");
			String attrType = Attribute.typeToString( df.classAttribute() );
			script.append("args['class_type'] = '" + attrType + "'\n");
		}
		//script.append("args['num_attributes'] = " + (df.numAttributes()-1) + "\n");
		//script.append("args['num_instances'] = " + df.numInstances() + "\n");
		script.append("args['relation_name'] = " +
	    		"'" + df.relationName().replace("'", "") + "'" + "\n");
	    
	    // pass attribute information
	    StringBuilder attrNames = new StringBuilder("args['attributes'] = [");
	    for(int i = 0; i < df.numAttributes(); i++) {
	    	String attrName = df.attribute(i).name();
	    	attrName = attrName.replace("'", "").replace("\"", "");
	    	attrNames.append( "'" + attrName + "'" );
	    	if(i != df.numAttributes()-1) {
	    		attrNames.append(",");
	    	}
	    }
	    attrNames.append("]\n");
	    script.append(attrNames.toString());
	    
	    HashMap<String, ArrayList<String>> m_attrEnums = new HashMap<String, ArrayList<String> >();
	    for(int i = 0; i < df.numAttributes(); i++) {
	    	if( df.attribute(i).isNominal() || df.attribute(i).isString() ) {
		    	Enumeration<Object> en = df.attribute(i).enumerateValues();
		    	ArrayList<String> strs = new ArrayList<String>(df.attribute(i).numValues());
		    	while(en.hasMoreElements()) {
		    		strs.add( escape( (String) en.nextElement() ) );
		    	}    	
		    	m_attrEnums.put(df.attribute(i).name(), strs);
	    	}
	    }
	    
	    // pass attribute enums
	    StringBuilder attrValues = new StringBuilder("args['attr_values'] = dict()\n");
	    for(String key : m_attrEnums.keySet()) {
	    	StringBuilder vector = new StringBuilder();
	    	vector.append("[");
	    	ArrayList<String> vals = m_attrEnums.get(key);
	    	for(String val : vals) {
	    		vector.append( "'" + val + "'" + "," );
	    	}
	    	vector.append("]");
	    	attrValues.append("args['attr_values']['" + escape(key) + "'] = " + vector.toString() + "\n" );
	    }
	    //session.executeScript(attrValues.toString(), debug);
	    script.append(attrValues.toString());
	    
	    // pass class name
	    if(df.classIndex() != -1) {
	    	String classAttr = escape(df.classAttribute().name());
	    	script.append( "args['class'] = '" + classAttr + "'\n");
	    	script.append( "args['class_index'] = " + df.classIndex() + "\n");
	    }
	    
	    // pass attribute types
	    StringBuilder attrTypes = new StringBuilder("args['attr_types'] = dict()\n");
	    for(int i = 0; i < df.numAttributes(); i++) {
	    	String attrName = df.attribute(i).name();
	    	String attrType = Attribute.typeToString( df.attribute(i) );
	    	attrTypes.append( "args['attr_types']['" + escape(attrName) + "'] = '" + attrType + "'\n" );
	    }
	    script.append( attrTypes.toString() );
	    
	    // custom arguments
	    if( !customParams.equals("") ) {
		    String[] extraParams = customParams.split(";");
		    for(String param : extraParams) {
		    	String[] paramSplit = param.split("=");
		    	script.append("args['" + paramSplit[0] + "'] = " + paramSplit[1] + "\n");
		    }
	    }
	    
	    return script.toString();
	    
	}
	
	public static File tempFileFromString(String script) throws Exception {
		File tmp = File.createTempFile("pyscript", ".py");
		PrintWriter pw = new PrintWriter(tmp);
		pw.write( script );
		pw.flush();
		pw.close();
		return tmp;
	}

}
