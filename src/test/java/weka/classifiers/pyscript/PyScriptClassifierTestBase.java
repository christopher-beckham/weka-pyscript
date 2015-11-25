package weka.classifiers.pyscript;

import java.io.File;

import weka.classifiers.AbstractClassifierTest;
import weka.classifiers.Classifier;

public class PyScriptClassifierTestBase extends AbstractClassifierTest {

	public PyScriptClassifierTestBase(String name) {
		super(name);
	}

	@Override
	public Classifier getClassifier() {
		PyScriptClassifier ps = new PyScriptClassifier();
		ps.setPythonFile(new File("tests/zeror-all-class-types.py"));
		ps.setSaveScript(true);
		return ps;
	}

}
