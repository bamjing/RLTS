import pandas as pd

class PortfolioManager():
    def __init__(self, initialFunds, marginRequirement=3100, fee=2.36, spread=0.01, maintainceFunds = 2000, multiplier=1000):
        self.funds = initialFunds
        self.index = 0
        self.transactionHistory = pd.DataFrame()
        self.activeTransaction = pd.DataFrame()
        self.MARGINREQUIREMENT = marginRequirement
        self.MAINTAINCEFUNDS = maintainceFunds
        self.MUTIPLIER = multiplier
        self.FEE = fee
        self.SPREAD = spread

        self.directional_rewards = 0

    def get_position(self):
        pos = 0
        if len(self.activeTransaction) != 0:
            pos = self.activeTransaction["Position"][0]
        return pos

    def proceedAction(self, action, currentState):
        ########################################################################################################
        date = currentState["Date"]
        actPrice = currentState["Open"]
        resultPrice = currentState["Close"]

        realizedFunds = 0
        anticipatedRewards = 0
        ########################################################################################################

        if action != 0:

            actionFee = self.FEE * abs(action)

            self.directional_rewards = (resultPrice - actPrice) * action

            ########################################################################################################
            currentTransaction = pd.DataFrame({'PurchaseDate': date, 'Position': action, 'Price': actPrice, 'Fee': actionFee},
                                              index=[self.index], columns=['PurchaseDate', 'Position', 'Price', 'Fee'])
            self.transactionHistory = pd.concat([self.transactionHistory, currentTransaction])
            self.index += 1
            ########################################################################################################

            if len(self.activeTransaction) == 0:

                self.activeTransaction = pd.DataFrame(
                    {'RecentDate': date, 'Position': action, 'Price': actPrice, 'Fee': actionFee},
                    index=[0], columns=['RecentDate', 'Position', 'Price', 'Fee'])

                anticipatedRewards = (resultPrice - actPrice) * action * self.MUTIPLIER - self._getSpreadCost(action) - actionFee * 2
            else:
                historicalAveragePurchasedPrice = self.activeTransaction["Price"][0]
                historicalPosition = self.activeTransaction["Position"][0]
                historicalFee = self.activeTransaction["Fee"][0]

                self.activeTransaction = self.activeTransaction[0: 0]

                ########################################################################################################

                if (historicalPosition / abs(historicalPosition)) * (action / abs(action)) == 1:
                    newPosition = historicalPosition + action
                    newFee = historicalFee + actionFee

                    newAveragePurchasedPrice = (historicalAveragePurchasedPrice * historicalPosition + actPrice * action) / newPosition
                    activeCurrentTransaction = pd.DataFrame({'RecentDate': date,
                                                             'Position': newPosition,
                                                             'Price': newAveragePurchasedPrice,
                                                             'Fee': newFee},
                                                            index=[0], columns=['RecentDate', 'Position', 'Price', 'Fee'])
                    self.activeTransaction = pd.concat([self.activeTransaction, activeCurrentTransaction])

                    anticipatedRewards = (resultPrice - newAveragePurchasedPrice) * newPosition * self.MUTIPLIER - self._getSpreadCost(newPosition) - newFee * 2
                else:
                    if abs(action) > abs(historicalPosition):
                        newPosition = action + historicalPosition
                        newFee = actionFee - historicalFee
                        activeCurrentTransaction = pd.DataFrame({'RecentDate': date,
                                                                 'Position': newPosition,
                                                                 'Price': actPrice,
                                                                 'Fee': newFee},
                                                                index=[0], columns=['RecentDate', 'Position', 'Price', 'Fee'])
                        self.activeTransaction = pd.concat([self.activeTransaction, activeCurrentTransaction])

                        anticipatedRewards = (resultPrice - actPrice) * newPosition * self.MUTIPLIER - self._getSpreadCost(newPosition) - newFee * 2
                        realizedFunds = (actPrice - historicalAveragePurchasedPrice) * historicalPosition * self.MUTIPLIER - self._getSpreadCost(historicalPosition) - historicalFee * 2
                        self.funds += realizedFunds

                    elif abs(action) == abs(historicalPosition):
                        realizedFunds = (actPrice - historicalAveragePurchasedPrice) * historicalPosition * self.MUTIPLIER - self._getSpreadCost(historicalPosition) - historicalFee * 2
                        self.funds += realizedFunds

                    else:
                        newPosition = action + historicalPosition
                        newFee = historicalFee - actionFee
                        activeCurrentTransaction = pd.DataFrame({'RecentDate': date,
                                                                 'Position': newPosition,
                                                                 'Price': historicalAveragePurchasedPrice,
                                                                 'Fee': newFee},
                                                                index=[0], columns=['RecentDate', 'Position', 'Price', 'Fee'])
                        self.activeTransaction = pd.concat([self.activeTransaction, activeCurrentTransaction])

                        anticipatedRewards = (resultPrice - historicalAveragePurchasedPrice) * newPosition * self.MUTIPLIER - self._getSpreadCost(newPosition) - newFee * 2
                        realizedFunds = -(actPrice - historicalAveragePurchasedPrice) * action * self.MUTIPLIER - self._getSpreadCost(action) - actionFee * 2
                        self.funds += realizedFunds

        else:

            self.directional_rewards = 0

            if len(self.activeTransaction) == 0:
                anticipatedRewards = 0
                realizedFunds = 0
            else:
                historicalAveragePurchasedPrice = self.activeTransaction["Price"][0]
                historicalPosition = self.activeTransaction["Position"][0]
                historicalFee = self.activeTransaction["Fee"][0]

                anticipatedRewards = (resultPrice - historicalAveragePurchasedPrice) * historicalPosition * self.MUTIPLIER - self._getSpreadCost(historicalPosition) - historicalFee * 2

        return [self.funds, realizedFunds, anticipatedRewards], self.get_position()


    def nextAvailableActions(self, nextState):

        nextState =  nextState.iloc[-1]
        actPrice = nextState["Open"]

        if len(self.activeTransaction) == 0:
            availableStep = int(self.funds / self.MARGINREQUIREMENT)
            stepRange = range(-availableStep, availableStep + 1)

        else:

            historicalAveragePurchasedPrice = self.activeTransaction["Price"][0]
            historicalPosition = self.activeTransaction["Position"][0]
            historicalFee = self.activeTransaction["Fee"][0]

            longShortPrefix = historicalPosition / abs(historicalPosition)

            unrealizedFunds = (actPrice - historicalAveragePurchasedPrice) * historicalPosition * self.MUTIPLIER - historicalFee

            positionSide = self.funds - abs(historicalPosition) * self.MARGINREQUIREMENT + unrealizedFunds
            nonPositionSide = self.funds + unrealizedFunds

            pSideAction = int(longShortPrefix * positionSide / self.MARGINREQUIREMENT)
            nPSideAction = -int(longShortPrefix * nonPositionSide / self.MARGINREQUIREMENT)

            left = min(pSideAction, nPSideAction)
            right = max(pSideAction, nPSideAction)

            stepRange = range(left, right + 1)

        return stepRange


    def _getSpreadCost(self, position):
        return self.SPREAD * self.MUTIPLIER * abs(position)