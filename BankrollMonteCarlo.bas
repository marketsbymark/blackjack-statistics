Attribute VB_Name = "modBankrollMonteCarlo"
Option Explicit

' Compatibility entry points only. The old fixed-outcome model did not model
' bankroll-funded splits/doubles and is not an exact model of this card engine.
' Do not publish stale or mathematically incompatible results from that model.
Private Sub ExplainSupportedEngine()
    MsgBox "The legacy fixed-outcome VBA model has been retired. " & _
        "Run bankroll_sim.py or the Streamlit dashboard for the card-level engine, " & _
        "the $25 table minimum, funded splits/doubles, and current Excel exports. " & _
        "No workbook results have been updated.", vbExclamation
End Sub

Public Sub RunBankrollMonteCarloAndExact()
    ExplainSupportedEngine
End Sub

Public Sub RunBankrollMonteCarlo()
    ExplainSupportedEngine
End Sub

Public Sub RunExactSurvivalModel()
    ExplainSupportedEngine
End Sub
