% Class definition
classdef session
    properties
        Date % Property to store the date
        Year % Property to store the year
        Session % Property to store Session
        Subject % Subject Number
        Repetition %
        Type % MI vs ME vs EOG
        Online % Online vs Offline
        Filename % Filename
        Spectrum % Spectrum
        Famp % Frequencies
        PEData % PEData
    end
    
    methods
        function obj = Record(obj,date, year, session, subject, repetition, type, online)
            % Constructor to initialize the properties
            if nargin > 0
                obj.Date = date;
                obj.Year = year;
                obj.Session = session;
                obj.Subject = subject;
                obj.Repetition = repetition;
                obj.Type = type;
                obj.Online = online;
                obj.Filename = filename;
            end
        end
        function obj = Store(obj, pe_data, pe_spectrum, pe_famp)
            obj.Spectrum = pe_spectrum;
            obj.PEData = pe_data;
            obj.Famp = pe_famp;
        end
    end
end