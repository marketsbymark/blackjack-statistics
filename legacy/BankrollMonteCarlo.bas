Attribute VB_Name = "modBankrollMonteCarlo"
Option Explicit

Public Sub RunBankrollMonteCarloAndExact()
    RunBankrollMonteCarlo
    RunExactSurvivalModel
    MsgBox "Bankroll model complete. Dashboard should now reflect updated Simulation and Exact_Model outputs.", vbInformation
End Sub

Public Sub RunBankrollMonteCarlo()
    Dim wsIn As Worksheet, wsOut As Worksheet, wsSim As Worksheet
    Dim startBankroll As Double, baseBet As Double, secondsPerHand As Double
    Dim trials As Long, maxHands As Long, lastOutcomeRow As Long
    Dim probs() As Double, netDollars() As Double, cumProb() As Double
    Dim totalProb As Double, r As Double, bankroll As Double
    Dim outcomeCount As Long, i As Long, t As Long, h As Long, idx As Long
    Dim outArr() As Variant

    Set wsIn = ThisWorkbook.Worksheets("Inputs")
    Set wsOut = ThisWorkbook.Worksheets("Outcomes")
    Set wsSim = ThisWorkbook.Worksheets("Simulation")

    startBankroll = wsIn.Range("B4").Value
    baseBet = wsIn.Range("B5").Value
    secondsPerHand = wsIn.Range("B6").Value
    trials = CLng(wsIn.Range("B7").Value)
    maxHands = CLng(wsIn.Range("B8").Value)

    lastOutcomeRow = wsOut.Cells(wsOut.Rows.Count, "C").End(xlUp).Row
    outcomeCount = 0
    totalProb = 0

    For i = 7 To lastOutcomeRow
        If wsOut.Cells(i, "C").Value > 0 Then
            outcomeCount = outcomeCount + 1
            ReDim Preserve probs(1 To outcomeCount)
            ReDim Preserve netDollars(1 To outcomeCount)
            ReDim Preserve cumProb(1 To outcomeCount)
            probs(outcomeCount) = wsOut.Cells(i, "C").Value
            netDollars(outcomeCount) = wsOut.Cells(i, "B").Value * baseBet
            totalProb = totalProb + probs(outcomeCount)
            cumProb(outcomeCount) = totalProb
        End If
    Next i

    If Abs(totalProb - 1) > 0.000001 Then
        MsgBox "Outcome probabilities must sum to 100%. Current total = " & Format(totalProb, "0.0000%"), vbExclamation
        Exit Sub
    End If

    wsSim.Range("A4:F100000").ClearContents
    ReDim outArr(1 To trials, 1 To 6)

    Randomize
    Application.ScreenUpdating = False
    Application.Calculation = xlCalculationManual

    For t = 1 To trials
        bankroll = startBankroll
        For h = 1 To maxHands
            r = Rnd()
            For idx = 1 To outcomeCount
                If r <= cumProb(idx) Then
                    bankroll = bankroll + netDollars(idx)
                    Exit For
                End If
            Next idx
            If bankroll <= 0 Then
                bankroll = 0
                Exit For
            End If
        Next h

        outArr(t, 1) = t
        outArr(t, 2) = h
        outArr(t, 3) = h * secondsPerHand / 60
        outArr(t, 4) = h * secondsPerHand / 3600
        outArr(t, 5) = bankroll
        If bankroll <= 0 Then
            outArr(t, 6) = "Busted"
        Else
            outArr(t, 6) = "Still Alive at Max Hands"
        End If
    Next t

    wsSim.Range("A4").Resize(trials, 6).Value = outArr
    wsSim.Range("I12").Value = Now

    Application.Calculation = xlCalculationAutomatic
    Application.ScreenUpdating = True
End Sub

Public Sub RunExactSurvivalModel()
    Dim wsIn As Worksheet, wsOut As Worksheet, wsExact As Worksheet
    Dim startBankroll As Double, stepSize As Double, secondsPerHand As Double
    Dim maxHands As Long, maxState As Long, startState As Long
    Dim lastOutcomeRow As Long, outcomeCount As Long
    Dim moveSteps() As Long, probs() As Double
    Dim states() As Double, newStates() As Double, results() As Variant
    Dim i As Long, h As Long, s As Long, dest As Long, idx As Long
    Dim totalProb As Double, expectedBankroll As Double

    Set wsIn = ThisWorkbook.Worksheets("Inputs")
    Set wsOut = ThisWorkbook.Worksheets("Outcomes")
    Set wsExact = ThisWorkbook.Worksheets("Exact_Model")

    startBankroll = wsIn.Range("B4").Value
    stepSize = wsIn.Range("B9").Value
    secondsPerHand = wsIn.Range("B6").Value
    maxHands = CLng(wsIn.Range("B8").Value)

    maxState = Application.WorksheetFunction.Max(100, CLng(Application.WorksheetFunction.RoundUp((startBankroll * 5) / stepSize, 0)))
    startState = CLng(Application.WorksheetFunction.Round(startBankroll / stepSize, 0))

    lastOutcomeRow = wsOut.Cells(wsOut.Rows.Count, "C").End(xlUp).Row
    outcomeCount = 0
    totalProb = 0

    For i = 7 To lastOutcomeRow
        If wsOut.Cells(i, "C").Value > 0 Then
            outcomeCount = outcomeCount + 1
            ReDim Preserve moveSteps(1 To outcomeCount)
            ReDim Preserve probs(1 To outcomeCount)
            moveSteps(outcomeCount) = CLng(Application.WorksheetFunction.Round((wsOut.Cells(i, "B").Value * wsIn.Range("B5").Value) / stepSize, 0))
            probs(outcomeCount) = wsOut.Cells(i, "C").Value
            totalProb = totalProb + probs(outcomeCount)
        End If
    Next i

    If Abs(totalProb - 1) > 0.000001 Then
        MsgBox "Outcome probabilities must sum to 100%. Current total = " & Format(totalProb, "0.0000%"), vbExclamation
        Exit Sub
    End If

    ReDim states(0 To maxState)
    ReDim newStates(0 To maxState)
    ReDim results(1 To maxHands + 1, 1 To 5)
    states(startState) = 1

    For h = 0 To maxHands
        expectedBankroll = 0
        For s = 0 To maxState
            expectedBankroll = expectedBankroll + states(s) * s * stepSize
        Next s

        results(h + 1, 1) = h
        results(h + 1, 2) = h * secondsPerHand / 3600
        results(h + 1, 3) = states(0)
        results(h + 1, 4) = 1 - states(0)
        results(h + 1, 5) = expectedBankroll

        If h < maxHands Then
            For s = 0 To maxState
                newStates(s) = 0
            Next s
            newStates(0) = states(0)

            For s = 1 To maxState
                If states(s) > 0 Then
                    For idx = 1 To outcomeCount
                        dest = s + moveSteps(idx)
                        If dest <= 0 Then
                            newStates(0) = newStates(0) + states(s) * probs(idx)
                        ElseIf dest >= maxState Then
                            newStates(maxState) = newStates(maxState) + states(s) * probs(idx)
                        Else
                            newStates(dest) = newStates(dest) + states(s) * probs(idx)
                        End If
                    Next idx
                End If
            Next s

            For s = 0 To maxState
                states(s) = newStates(s)
            Next s
        End If
    Next h

    wsExact.Range("A4:E100000").ClearContents
    wsExact.Range("A3:E3").Value = Array("Hand", "Hours", "Bust Probability", "Survival Probability", "Expected Bankroll")
    wsExact.Range("A4").Resize(maxHands + 1, 5).Value = results
End Sub
