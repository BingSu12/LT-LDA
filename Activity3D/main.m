%% construct data for trainingLTLDA
construct_data_for_trainingLTLDA

%% calculate the transformation
Calculate_LTLDA_Transformation
% the learned transformation is saved in './datamat/'

%% perform classification using Kerlized-COV
% load the transformation in GenerateActionDataDisMatrix_At3D
run_me_KMAt3D