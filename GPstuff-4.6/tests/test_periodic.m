function test_suite = test_periodic

%   Run specific demo, save values and compare the results to the expected.
%   Works for both xUnit Test Framework package by Steve Eddins and for
%   the built-in Unit Testing Framework (as of Matlab version 2013b).
%
%   See also
%     TEST_ALL, DEMO_PERIODIC
%
% Copyright (c) 2011-2012 Ville Tolvanen
% Copyright (c) 2014 Tuomas Sivula

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.
  
  % Check which package to use
  if exist('initTestSuite', 'file')
    initTestSuite_path = which('initTestSuite');
  else
    initTestSuite_path = '';
  end
  if ~isempty(initTestSuite_path) && ...
     exist([initTestSuite_path(1:end-15) 'runtests'], 'file')
    % xUnit package
    initTestSuite;
  else
    % Built-in package
    % Use all functions except the @setup
    tests = localfunctions;
    tests = tests(~cellfun(@(x)strcmp(func2str(x),'setup'),tests));
    test_suite = functiontests(tests);
  end
end


% -------------
%     Tests
% -------------

function testRunDemo(testCase)
  % Run the correspondin demo and save the values. Note this test has to
  % be run at lest once before the other test may succeed.
  rundemo(getName())
end

function testPredictionsMaunaLoa(testCase)
  verifyVarsEqual(testCase, getName(), ...
    {'Eft_full1', 'Eft_full2','Varft_full1','Varft_full2'}, @mean, ...
    'RelTolElement', 0.05, 'RelTolRange', 0.01)
end

function testPredictionsDrowning(testCase)
  verifyVarsEqual(testCase, getName(), ...
    {'Eft_full', 'Varft_full'}, @mean, ...
    'RelTolElement', 0.05, 'RelTolRange', 0.01)
end


% ------------------------
%     Helper functions
% ------------------------

function testCase = setup
  % Helper function to suply empty array into variable testCase as an
  % argument for each test function, if using xUnit package. Not to be
  % used with built-in test framework.
  testCase = [];
end

function name = getName
  % Helperfunction that returns the name of the demo, e.g. 'binomial1'.
  name = mfilename;
  name = name(6:end);
end

