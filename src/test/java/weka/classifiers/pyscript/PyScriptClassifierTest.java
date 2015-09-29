package weka.classifiers.pyscript;



import static org.junit.Assert.*;

import java.io.File;

import org.junit.Test;

import weka.classifiers.AbstractClassifierTest;
import weka.classifiers.Classifier;
import weka.core.BatchPredictor;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class PyScriptClassifierTest {
	
	//public PyScriptClassifierTest(String name) {
	//	super(name);
	//}

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
		assertNotEquals(ps.getModelString(), null);
		assertNotEquals( ps.distributionsForInstances(train), null );
	}
	
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
		assertNotEquals(ps.getModelString(), null);
		assertNotEquals( ps.distributionsForInstances(train), null );
	}
	
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
		assertNotEquals(ps.getModelString(), null);
		assertNotEquals( ps.distributionsForInstances(train), null );	
	}
	
	@Test
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

	public Classifier getClassifier() {
		PyScriptClassifier ps = new PyScriptClassifier();
		System.out.println( ps.getBatchSize() );
		ps.setDebug(true);
		return ps;
	}

}
