package weka.classifiers.pyscript;

import static org.junit.Assert.*;

import java.io.File;
import org.junit.Test;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class PyScriptClassifierTest {
	
	/**
	 * Test to see if the classifier will work without error
	 * for a peculiar ARFF file.
	 * @throws Exception
	 */
	@Test
	public void testSpecialCharacters() throws Exception {
		PyScriptClassifier ps = (PyScriptClassifier) getClassifier();
		ps.setPythonFile( new File("scripts/zeror.py") );
		DataSource ds = new DataSource("datasets/special-chars.arff");
		Instances data = ds.getDataSet();
		data.setClassIndex( data.numAttributes() - 1);
		ps.buildClassifier(data);
	}

	/**
	 * Not testing anything in particular here - just make sure
	 * that we can train the RF example without some
	 * exception being thrown.
	 * @throws Exception
	 */
	@Test
	public void testRandomForestOnDiabetes() throws Exception {
		System.out.println("testRandomForestOnDiabetes()");
		PyScriptClassifier ps = (PyScriptClassifier) getClassifier();
		ps.setPythonFile( new File("scripts/scikit-rf.py") );
		ps.setArguments("num_trees=10");
		DataSource ds = new DataSource("datasets/diabetes.arff");
		Instances train = ds.getDataSet();
		train.setClassIndex( train.numAttributes() - 1 );
		ps.buildClassifier(train);
	}
	
	/**
	 * Not testing anything in particular here - just make sure
	 * that we can train the linear reg example without some
	 * exception being thrown.
	 * @throws Exception
	 */
	@Test
	public void testLinearRegressionOnDiabetes() throws Exception {
		System.out.println("testLinearRegressionOnDiabetes()");
		PyScriptClassifier ps = (PyScriptClassifier) getClassifier();
		ps.setPythonFile( new File("scripts/linear-reg.py") );
		ps.setArguments("alpha=0.01;epsilon=0.0001");
		DataSource ds = new DataSource("datasets/diabetes_numeric.arff");
		Instances train = ds.getDataSet();
		train.setClassIndex( train.numAttributes() - 1 );
		ps.setShouldStandardize(true);
		ps.buildClassifier(train);
	}
	
	/**
	 * Not testing anything in particular here - just make sure
	 * that we can train ZeroR on Iris without some exception
	 * being thrown.
	 * @throws Exception
	 */
	@Test
	public void testZeroROnIris() throws Exception {
		System.out.println("testZeroROnIris()");
		PyScriptClassifier ps = (PyScriptClassifier) getClassifier();
		ps.setPythonFile( new File("scripts/zeror.py") );
		ps.setArguments("");
		DataSource ds = new DataSource("datasets/iris.arff");
		Instances train = ds.getDataSet();
		train.setClassIndex( train.numAttributes() - 1 );
		ps.buildClassifier(train);
	}
	
	/**
	 * Test to see an exception gets thrown when a "bad"
	 * script is given to the classifier.
	 * @throws Exception
	 */
	@Test(expected=Exception.class)
	public void testExceptionRaiser() throws Exception {
		System.out.println("testExceptionRaiser()");
		PyScriptClassifier ps = (PyScriptClassifier) getClassifier();
		ps.setDebug(true);
		ps.setPythonFile( new File("scripts/test-exception.py") );
		DataSource ds = new DataSource("datasets/iris.arff");
		Instances train = ds.getDataSet();
		ps.buildClassifier(train);
		assertEquals(ps.getModelString(), null);
		
	}
	
	/**
	 * Test to see if the script save feature works.
	 * @throws Exception
	 */
	@Test
	public void testScriptSave() throws Exception {
		System.out.println("testScriptSave()");
		PyScriptClassifier ps = (PyScriptClassifier) getClassifier();
		ps.setDebug(true);
		ps.setPythonFile( new File("scripts/zeror.py") );
		ps.setArguments("");
		DataSource ds = new DataSource("datasets/iris.arff");
		Instances train = ds.getDataSet();
		train.setClassIndex( train.numAttributes() - 1 );
		ps.setSaveScript(true);
		ps.buildClassifier(train);
		// we saved the script so it doesn't matter where it is now
		ps.setPythonFile( new File("bad-file.py") );
		ps.distributionsForInstances(train);
	}

	public Classifier getClassifier() {
		PyScriptClassifier ps = new PyScriptClassifier();
		System.out.println( ps.getBatchSize() );
		ps.setDebug(true);
		return ps;
	}

}
