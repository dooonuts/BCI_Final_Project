function fisher_scores = fisher_score(X, y)
    % X: Feature matrix (n_samples x n_features)
    % y: Labels (n_samples)
    [n_samples, n_features] = size(X); % Get the number of samples and features
    classes = unique(y); % Get the unique classes
    fisher_scores = zeros(1, n_features); % Initialize an array to store the Fisher scores
    % Compute Fisher score for each feature
    for f = 1:n_features
        feature_values = X(:, f); % Get the values of the f-th feature
        between_class_variance = 0;
        within_class_variance = 0;
        % Compute the overall mean of the feature
        mu = mean(feature_values);
        % Loop through each class to compute the variances
        for c = 1:length(classes)
            class_samples = feature_values(y == classes(c)); % Get samples of class c
            mu_c = mean(class_samples); % Mean of the feature for class c
            n_c = length(class_samples); % Number of samples in class c
            % Between-class variance
            between_class_variance = between_class_variance + n_c * (mu_c - mu)^2;
            % Within-class variance
            within_class_variance = within_class_variance + sum((class_samples - mu_c).^2);
        end
        % Compute the Fisher score for this feature
        fisher_scores(f) = between_class_variance / within_class_variance;
    end
end