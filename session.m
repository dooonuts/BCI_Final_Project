% Class definition
classdef session
    properties
        Date % Property to store the date
        Year % Property to store the year
    end
    
    methods
        function obj = Record(date, year)
            % Constructor to initialize the properties
            if nargin > 0
                obj.Date = date;
                obj.Year = year;
            end
        end
    end
end