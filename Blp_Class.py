import blpapi
import pandas as pd
import datetime as dt

DATE = blpapi.Name("date")
ERROR_INFO = blpapi.Name("errorInfo")
EVENT_TIME = blpapi.Name("EVENT_TIME")
FIELD_DATA = blpapi.Name("fieldData")
FIELD_EXCEPTIONS = blpapi.Name("fieldExceptions")
FIELD_ID = blpapi.Name("fieldId")
SECURITY = blpapi.Name("security")
SECURITY_DATA = blpapi.Name("securityData")


class BLP:
    # -----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------

    def __init__(self):
        """
            Improve this
            BLP object initialization
            Synchronus event handling

        """
        # Create Session object
        self.session = blpapi.Session()

        # Exit if can't start the Session
        if not self.session.start():
            print("Failed to start session.")
            return

        # Open & Get RefData Service or exit if impossible
        if not self.session.openService("//blp/refdata"):
            print("Failed to open //blp/refdata")
            return

        self.session.openService('//BLP/refdata')
        self.refDataSvc = self.session.getService('//BLP/refdata')

        print('Session open')

    # -----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------

    def bdh(self, strSecurity, strFields, startdate, enddate, per='DAILY', perAdj='CALENDAR',
            days='NON_TRADING_WEEKDAYS', fill='PREVIOUS_VALUE', curr=None):
        """
            Summary:
                HistoricalDataRequest ;

                Gets historical data for a set of securities and fields

            Inputs:
                strSecurity: list of str : list of tickers
                strFields: list of str : list of fields, must be static fields (e.g. px_last instead of last_price)
                startdate: date
                enddate
                per: periodicitySelection; daily, monthly, quarterly, semiannually or annually
                perAdj: periodicityAdjustment: ACTUAL, CALENDAR, FISCAL
                curr: string, else default currency is used
                Days: nonTradingDayFillOption : NON_TRADING_WEEKDAYS*, ALL_CALENDAR_DAYS or ACTIVE_DAYS_ONLY
                fill: nonTradingDayFillMethod :  PREVIOUS_VALUE, NIL_VALUE

                Options can be selected these are outlined in “Reference Services and Schemas Guide.”

            Output:
                A list containing as many dataframes as requested fields
            # Partial response : 6
            # Response : 5

        """

        # -----------------------------------------------------------------------
        # Create request
        # -----------------------------------------------------------------------

        # Create request
        request = self.refDataSvc.createRequest('HistoricalDataRequest')

        # Put field and securities in list is single value is passed
        if type(strFields) == str:
            strFields = [strFields]

        if type(strSecurity) == str:
            strSecurity = [strSecurity]

        # Append list of securities/ On rajoute les fields à la requette
        for strF in strFields:
            request.append('fields', strF)

        for strS in strSecurity:
            request.append('securities', strS)

        # Set other parameters. On converti en string les dates
        request.set('startDate', startdate.strftime('%Y%m%d'))
        request.set('endDate', enddate.strftime('%Y%m%d'))
        request.set('periodicitySelection', per)

        # Set other parameters. currency days/ per adjustement
        request.set('nonTradingDayFillOption', days)
        if curr != None:
            request.set('currency', curr)
        request.set('periodicityAdjustment', perAdj)
        request.set('nonTradingDayFillMethod', fill)
        # -----------------------------------------------------------------------
        # Send request
        # -----------------------------------------------------------------------

        requestID = self.session.sendRequest(request)
        print("Sending BDH request")

        # -----------------------------------------------------------------------
        # Receive request
        # -----------------------------------------------------------------------

        list_msg = []
        # Create the variable

        while True:
            event = self.session.nextEvent()

            # Ignores anything that's not partial or final
            if (event.eventType() != blpapi.event.Event.RESPONSE) & (
                    event.eventType() != blpapi.event.Event.PARTIAL_RESPONSE):
                # itération suivante de la boucle
                continue

            # Extract the response message
            msg = blpapi.event.MessageIterator(event).__next__()
            list_msg.append(msg)


            # Break loop if response is final (response = fini c'était le dernier message)
            if event.eventType() == blpapi.event.Event.RESPONSE:
                break

                # -----------------------------------------------------------------------
        # Exploit data
        # -----------------------------------------------------------------------

        dict_Fields_Dataframe = {}

        for fieldd in strFields:
            globals()['dict_' + fieldd] = {}

        for msg in list_msg:

            secur_data = msg.getElement(SECURITY_DATA)
            # -------------------------------------------------------------------------------------
            # Partie en bleu
            securityName = str(secur_data.getElement(SECURITY).getValue())
            # Nom de la security
            field_data = secur_data.getElement(FIELD_DATA)

            for fieldd in strFields:
                globals()['dict_' + fieldd][securityName] = {}

            # nombre de date
            int_nbDate = field_data.numValues()

            for i in range(0, int_nbDate):

                fields = field_data.getValue(i)
                nb_fields = fields.numElements()

                dt_date = fields.getElement(0).getValue()

                for j in range(1, nb_fields):
                    element = fields.getElement(j)
                    field_name = str(element.name())  # Volume/ px_lasts
                    field_value = element.getValue()

                    globals()['dict_' + field_name][securityName][dt_date] = field_value


        # pandas.dataframe.from.dict( , )
        for field in strFields:
            dict_Fields_Dataframe[field] = pd.DataFrame.from_dict(globals()['dict_' + field], orient='columns')
        return dict_Fields_Dataframe

    # -----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------
    def bds(self, strSecurity, strFields, strOverrideField='', strOverrideValue=''):

        """
            Summary:
                Reference Data Request ; Real-time if entitled, else delayed values
                Only supports 1 override


            Input:
                strSecurity
                strFields
                strOverrideField
                strOverrideValue

            Output:
               Dict
        """

        # -----------------------------------------------------------------------
        # Create request
        # -----------------------------------------------------------------------

        # Create request
        request = self.refDataSvc.createRequest('ReferenceDataRequest')

        # Put field and securities in list is single field passed
        if type(strFields) == str:
            strFields = [strFields]

        if type(strSecurity) == str:
            strSecurity = [strSecurity]

        # Append list of fields
        for strD in strFields:
            request.append('fields', strD)

        # Append list of securities
        for strS in strSecurity:
            request.append('securities', strS)

        # Add override
        if strOverrideField != '':
            o = request.getElement('overrides').appendElement()
            o.setElement('fieldId', strOverrideField)
            o.setElement('value', strOverrideValue)

        # -----------------------------------------------------------------------
        # Send request
        # -----------------------------------------------------------------------
        requestID = self.session.sendRequest(request)
        print("Sending BDS request")

        # -----------------------------------------------------------------------
        # Receive request
        # -----------------------------------------------------------------------
        list_msg = []
        dict_Fields_Dataframe = {}
        for fieldd in strFields:
            globals()['dict_' + fieldd] = {}

        while True:
            event = self.session.nextEvent()

            # Ignores anything that's not partial or final
            if (event.eventType() != blpapi.event.Event.RESPONSE) & (
                    event.eventType() != blpapi.event.Event.PARTIAL_RESPONSE):
                continue

            # Extract the response message
            msg = blpapi.event.MessageIterator(event).__next__()
            list_msg.append(msg)
            # Break loop if response is final
            if event.eventType() == blpapi.event.Event.RESPONSE:
                break

        for msg in list_msg:
            secur_data = msg.getElement(SECURITY_DATA)
            nb_securities = secur_data.numValues()

            for i in range(0, nb_securities):
                security_data = secur_data.getValue(i)
                security_name = security_data.getElement(SECURITY).getValue()
                field_data = security_data.getElement(FIELD_DATA)
                nb_values = field_data.numElements()

                for fieldd in strFields:
                    globals()['dict_' + str(fieldd)][security_name] = {}

                for j in range(0, nb_values):
                    element = field_data.getElement(j)
                    field_name = str(element.name())
                    nb_element = element.numValues()

                    for i_ticker in range(nb_element):
                        ticker = element.getValue(i_ticker)
                        ticker_name = ticker.getElement(0).getValue()
                        field_value = ticker.getElement(1).getValue()

                        globals()['dict_' + str(field_name)][security_name][ticker_name] = field_value

            for field in strFields:
                # Fill the output dictionnary and adjust the data format
                if len(strFields) == 1 and nb_securities == 1:
                    dict_Fields_Dataframe = pd.DataFrame([globals()['dict_' + field]], index=[dt.datetime.now()]).iloc[
                        0, 0]

                if len(strFields) > 1 and nb_securities == 1:
                    dict_Fields_Dataframe[field] = pd.DataFrame.from_dict(globals()['dict_' + field])

                if len(strFields) == 1 and nb_securities > 1:
                    dict_Fields_Dataframe = pd.DataFrame([globals()['dict_' + field]], index=[dt.datetime.now()])

                if len(strFields) > 1 and nb_securities > 1:
                    dict_Fields_Dataframe[field] = pd.DataFrame([globals()['dict_' + field]], index=[dt.datetime.now()])
        return dict_Fields_Dataframe

        # Try exept pour récupere les valeurs en tant que float sinon STR
        # -----------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------------

    def bdp(self, strSecurity, strFields, strOverrideField='', strOverrideValue=''):

        """
            Summary:
                Reference Data Request ; Real-time if entitled, else delayed values
                Only supports 1 override


            Input:
                strSecurity
                strFields
                strOverrideField
                strOverrideValue

            Output:
               Dict
        """

        # -----------------------------------------------------------------------
        # Create request
        # -----------------------------------------------------------------------

        # Create request
        request = self.refDataSvc.createRequest('ReferenceDataRequest')

        # Put field and securities in list is single field passed
        if type(strFields) == str:
            strFields = [strFields]

        if type(strSecurity) == str:
            strSecurity = [strSecurity]

        # Append list of fields
        for strD in strFields:
            request.append('fields', strD)

        # Append list of securities
        for strS in strSecurity:
            request.append('securities', strS)

        # Add override
        if strOverrideField != '':
            o = request.getElement('overrides').appendElement()
            o.setElement('fieldId', strOverrideField)
            o.setElement('value', strOverrideValue)

        # -----------------------------------------------------------------------
        # Send request
        # -----------------------------------------------------------------------

        requestID = self.session.sendRequest(request)
        print("Sending BDP request")

        # -----------------------------------------------------------------------
        # Receive request
        # -----------------------------------------------------------------------

        list_msg = []
        dict_Fields_Dataframe = {}
        for fieldd in strFields:
            globals()['dict_' + fieldd] = {}

        while True:
            event = self.session.nextEvent()

            # Ignores anything that's not partial or final
            if (event.eventType() != blpapi.event.Event.RESPONSE) & (
                    event.eventType() != blpapi.event.Event.PARTIAL_RESPONSE):
                continue

            # Extract the response message
            msg = blpapi.event.MessageIterator(event).__next__()
            list_msg.append(msg)
            # Break loop if response is final
            if event.eventType() == blpapi.event.Event.RESPONSE:
                break

        # -----------------------------------------------------------------------
        # Extract the data
        # -----------------------------------------------------------------------

        for msg in list_msg:
            secur_data = msg.getElement(SECURITY_DATA)
            nb_ticker = secur_data.numValues()

            for i in range(0, nb_ticker):
                security_data = secur_data.getValue(i)
                security_name = security_data.getElement(SECURITY).getValue()
                field_data = security_data.getElement(FIELD_DATA)
                nb_values = field_data.numElements()

                for fieldd in strFields:
                    globals()['dict_' + str(fieldd)][security_name] = {}

                for j in range(0, nb_values):
                    element = field_data.getElement(j)
                    field_value = element.getValue()
                    field_name = str(element.name())

                    globals()['dict_' + str(field_name)][security_name] = field_value

        for field in strFields:
            # print(globals()['dict_' + field])
            dict_Fields_Dataframe[field] = pd.DataFrame([globals()['dict_' + field]], index=[dt.datetime.now()])

        return dict_Fields_Dataframe if len(strFields) > 1 else dict_Fields_Dataframe[field]

    # Try exept pour récupere les valeurs en tant que float sinon STR
    # -----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------

    def closeSession(self):
        print("Session closed")
        self.session.stop()

if __name__ == '__main__':
    blp = BLP()
    strFields = ["PX_LAST"]
    tickers = ["ATO FP Equity", "TTE FP Equity"]
    startDate = dt.datetime(2020, 10, 1)
    endDate = dt.datetime(2020, 11, 3)
    prices = blp.bdh(strSecurity=tickers, strFields=strFields, startdate=startDate, enddate=endDate)
    aa = prices['PX_LAST']
    print(prices)
