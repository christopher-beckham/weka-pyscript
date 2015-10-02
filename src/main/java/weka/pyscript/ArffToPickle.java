package weka.pyscript;

import java.util.List;

import weka.core.CommandlineRunnable;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.pyscript.Utility;
import weka.python.PythonSession;

/**
 * Convert an .arff file to a pkl.gz for the purposes of
 * testing Python classifiers independently of WEKA.
 * @author cjb60
 */
public class ArffToPickle implements CommandlineRunnable {
	
	private static final String USAGE = "Usage: -i <arff file> -o <destination file> -c <class index> "
			+ "[-cmd <python command>] [-args <string>] [-impute] [-binarize] [-standardize] [-debug]";
	
	private PythonSession m_session = null;
	private boolean m_debug = false;
	
	private String m_cmd = "python";
	private String m_filename = null;
	private String m_dest = null;
	
	private String m_classIndex = "last";
	
	private boolean m_shouldImpute = false;
	private boolean m_shouldStandardize = false;
	private boolean m_shouldBinarize = false;

	private String m_args = "";
	
	public void setArgs(String s) {
		m_args = s;
	}
	
	public String getArgs() {
		return m_args;
	}
	
	public void setPythonCommand(String s) {
		m_cmd = s;
	}
	
	public String getPythonCommand() {
		return m_cmd;
	}
	
	public void setFilename(String s) {
		m_filename = s;
	}
	
	public void setDest(String s) {
		m_dest = s;
	}
	
	public boolean getDebug() {
		return m_debug;
	}
	
	public void setDebug(boolean b) {
		m_debug = b;
	}
	
	public String getClassIndex() {
		return m_classIndex;
	}
	
	public void setClassIndex(String c) {
		m_classIndex = c;
	}
	
	public boolean getShouldImpute() {
		return m_shouldImpute;
	}
	
	public void setShouldImpute(boolean b) {
		m_shouldImpute = b;
	}
	
	public boolean getShouldStandardize() {
		return m_shouldStandardize;
	}
	
	public void setShouldStandardize(boolean b) {
		m_shouldStandardize = b;
	}
	
	public boolean getShouldBinarize() {
		return m_shouldBinarize;
	}
	
	public void setShouldBinarize(boolean b) {
		m_shouldBinarize = b;
	}
	
	public void setOptions(String[] options) {
		try {
			setFilename( Utils.getOption("i", options) );
			setDest( Utils.getOption("o", options) );
			setClassIndex( Utils.getOption("c", options) );
			
			String cmd = Utils.getOption("cmd", options);
			if( cmd.length() != 0) {
				setPythonCommand(cmd);
			}
			
			String args = Utils.getOption("args", options);
			setArgs(args);
			
			setShouldImpute( Utils.getFlag("impute", options) );
			setShouldBinarize( Utils.getFlag("binarize", options) );
			setShouldStandardize( Utils.getFlag("standardize", options) );
			
			setDebug( Utils.getFlag("debug", options) );
		} catch(Exception ex) {
			System.err.println(USAGE);
			ex.printStackTrace();
		}
		
	}
	
	public void convert() {
		try {
			m_session = Utility.initPythonSession(this, getPythonCommand(), m_debug);
			
			DataSource ds = new DataSource(m_filename);
			Instances instances = ds.getDataSet();
			try {
				if( getClassIndex().equals("first")) {
					instances.setClassIndex(0);
				} else if( getClassIndex().equals("last")) {
					instances.setClassIndex( instances.numAttributes() - 1 );
				} else {
					int classIdx = Integer.parseInt(getClassIndex());
					instances.setClassIndex(classIdx);
				}		
			} catch(NumberFormatException ex) {
				System.err.println("Illegal class index: " + getClassIndex());
				System.err.println("Assuming class index is 'last'");
				instances.setClassIndex( instances.numAttributes() - 1 );
			}
			
			instances = Utility.preProcessData(instances, 
					getShouldImpute(), getShouldBinarize(), getShouldStandardize() );
			
			if(m_classIndex.equals("first")) {
				instances.setClassIndex(0);
			} else if (m_classIndex.equals("last") ) {
				instances.setClassIndex( instances.numAttributes() - 1 );
			} else {
				instances.setClassIndex( Integer.parseInt(m_classIndex) );
			}
			
			List<String> out = m_session.executeScript(
				Utility.createArgsScript(instances, getArgs(), m_session, m_debug),
				m_debug
			);
		    if(out.get(1).contains(Utility.TRACEBACK_MSG)) {
		    	throw new Exception( "An error happened while trying to create the args variable:\n" + out.get(1) );
		    }
			
		    m_session.instancesToPythonAsScikitLearn(instances, "train", false);
		    m_session.executeScript("args['X_train'] = X\nargs['y_train'] = Y\n", getDebug());
			
	    	StringBuilder sb = new StringBuilder();
	    	sb.append("import gzip\n");
	    	sb.append("try:\n");
	    	sb.append("  import cPickle as pickle\n");
	    	sb.append("except ImportError:\n");
	    	sb.append("  import pickle\n");
	    	sb.append("_g = gzip.open('" + m_dest.replace("'","\\'") + "', 'wb')\n");
	    	sb.append("pickle.dump(args, _g, pickle.HIGHEST_PROTOCOL)\n");
	    	sb.append("_g.close()\n");
	    	m_session.executeScript(sb.toString(), m_debug);
	    	
		} catch(Exception ex) {
			System.err.println(USAGE);
			ex.printStackTrace();
		} finally {
			Utility.closePythonSession(this);
		}
	}

	@Override
	public void run(Object toRun, String[] options) {
		ArffToPickle arffToPickle = (ArffToPickle) toRun;
		arffToPickle.setOptions(options);
		arffToPickle.convert();
	}
	
	public static void main(String[] args) {
		new ArffToPickle().run(new ArffToPickle(), args);
	}
}
