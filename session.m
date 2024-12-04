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
        Label % 
        % Spectrum % Spectrum
        % Famp % Frequencies
        % PEData % PEData
        PE_MI % Processed MI Data
        MI_Tags % MI Tags
        PE_MI_Spectrum % Processed MI Spectrum
        PE_MI_Famp % Processed MI Frequency Amplitude
        PE_Rest % Processed Rest Data
        Rest_Tags % Rest Tags
        PE_Rest_Spectrum % Processed Rest Spectrum
        PE_Rest_Famp % Processed Rest Frequency Amplitude
        Raw_Data
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
        % function obj = Store(obj, pe_data, pe_spectrum, pe_famp)
        %     obj.Spectrum = pe_spectrum;
        %     obj.PEData = pe_data;
        %     obj.Famp = pe_famp;
        % end
    end
end