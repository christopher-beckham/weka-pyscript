package weka.classifiers.pyscript;

import weka.classifiers.AbstractClassifierTest;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class PyScriptClassifierTest extends AbstractClassifierTest {
	
	public PyScriptClassifierTest(String name) {
		super(name);
	}

	public void testRandomForestOnDiabetes() {
		try{
			PyScriptClassifier ps = (PyScriptClassifier) getClassifier();
			ps.setPythonFile("scripts/scikit-rf.py");
			ps.setTrainPythonFileParams("'num_trees'=10");
			ps.setTestPythonFileParams("'num_trees'=10");
			DataSource ds = new DataSource("datasets/diabetes.arff");
			Instances train = ds.getDataSet();
			train.setClassIndex( train.numAttributes() - 1 );
			ps.buildClassifier(train);
			ps.distributionsForInstances(train);
		} catch(Exception ex) {
			ex.printStackTrace();
			fail();
		}
	}
	
	public void testLinearRegressionOnDiabetes() {
		try {
			PyScriptClassifier ps = (PyScriptClassifier) getClassifier();
			ps.setPythonFile("scripts/linear-reg.py");
			ps.setTrainPythonFileParams("'alpha'=0.01,'epsilon'=0.0001");
			ps.setTestPythonFileParams("'alpha'=0.01,'epsilon'=0.0001");
			DataSource ds = new DataSource("datasets/diabetes_numeric.arff");
			Instances train = ds.getDataSet();
			train.setClassIndex( train.numAttributes() - 1 );
			ps.buildClassifier(train);
			ps.distributionsForInstances(train);
			
		} catch(Exception ex) {
			ex.printStackTrace();
			fail();
		}
	}

	@Override
	public Classifier getClassifier() {
		return new PyScriptClassifier();
	}

}
