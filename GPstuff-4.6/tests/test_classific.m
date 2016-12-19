function test_suite = test_classific

%   Run specific demo, save values and compare the results to the expected.
%   Works for both xUnit Test Framework package by Steve Eddins and for
%   the built-in Unit Testing Framework (as of Matlab version 2013b).
%
%   See also
%     TEST_ALL, DEMO_CLASSIFIC
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
  rundemo(getName(), {...
    {'Eft_la', @(x)x(1:100)}, ...
    {'Eft_ep', @(x)x(1:100)}, ...
    {'Efs_mc', @(x)x(1:50,1:50)}, ...
    {'Varft_la', @(x)x(1:100)}, ...
    {'Varft_ep', @(x)x(1:100)}, ...
    {'Varfs_mc', @(x)x(1:50,1:50)} })
end

function testLaplace(testCase)
  verifyVarsEqual(testCase, getName(), {'Eft_la','Varft_la'}, ...
    'RelTolElement', 0.05, 'RelTolRange', 0.01)
end

function testEP(testCase)
  verifyVarsEqual(testCase, getName(), {'Eft_ep','Varft_ep'}, ...
    'RelTolElement', 0.05, 'RelTolRange', 0.01)
end

% The results vary too much to be tested reliably
% function testMC(testCase)
%   verifyVarsEqual(testCase, getName(), {'Efs_mc'}, @(x)mean(x,2), ...
%     'RelTolRange', 0.2)
% end


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

