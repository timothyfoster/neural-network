classdef node
    properties
        weights
        bias
        sum
    end
    methods      
        function gn = node()
            gn.weights = [];
            gn.bias = 1;
            gn.sum = 0;
        end
    end
end