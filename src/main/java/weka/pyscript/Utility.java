package weka.pyscript;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;

import weka.core.Instances;
import weka.python.PythonSession;

/**
 * Helper functions for PyScript
 * @author cjb60
 *
 */
public class Utility {
	
	public static final String TRACEBACK_MSG = "Traceback (most recent call last):";
	
	/**
	 * Push an args variable to the specified Python
	 * session.
	 * @param df data frame
	 * @param session the Python session to send args to
	 * @param debug print debug information?
	 * @throws Exception if an error traceback has been detected in stderr
	 */
	public static void pushArgs(Instances df, String customParams,
			PythonSession session, boolean debug) throws Exception {
		
		StringBuilder script = new StringBuilder();
		
		script.append("args = dict()\n");
		
		// pass general information related to the training data
		if(df.classIndex() != -1) {
			script.append("args['num_classes'] = " + df.numClasses() + "\n");
		}
		script.append("args['num_attributes'] = " + df.numAttributes() + "\n");
		script.append("args['num_instances'] = " + df.numInstances() + "\n");
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
		    		strs.add( (String) en.nextElement() );
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
	    	attrValues.append("args['attr_values']['" + 
	    		key.replace("'", "\\'").replace("\n", "\\n") + "'] = " + vector.toString() + "\n" );
	    }
	    //session.executeScript(attrValues.toString(), debug);
	    script.append(attrValues.toString());
	    
	    // pass class name
	    if(df.classIndex() != -1) {
	    	String classAttr = df.classAttribute().name().replace("'", "").replace("\"", "");
	    	script.append( "args['class'] = '" + classAttr.replace("'", "") + "'\n");
	    }
	    
	    // pass custom parameters from -xp or -yp
	    /*
	    String customParams = null;
	    if(trainMode) {
	    	customParams = getTrainPythonFileParams();
	    } else {
	    	customParams = getTestPythonFileParams();
	    }
	    */
	    if( !customParams.equals("") ) {
		    String[] extraParams = customParams.split(",");
		    for(String param : extraParams) {
		    	String[] paramSplit = param.split("=");
		    	script.append("args[" + paramSplit[0] + "] = " + paramSplit[1]);
		    }
	    }
	    
	    List<String> out = session.executeScript(script.toString(), debug);
	    if(out.get(1).contains(TRACEBACK_MSG)) {
	    	throw new Exception( out.get(1) );
	    }
	    
	}

}
