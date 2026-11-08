> **Bu dosya ödevin orijinal şartnamesidir (değiştirilmeden saklanır).** Tüm tasarım kararları bu metne referans vermelidir. Türkçe özet için `01_PROJECT_OVERVIEW.md` dosyasına bakın.

# Project Specification: Football Pass Detection and Analysis System

## 1. Project Overview

Develop a computer vision-based football match analysis system that analyzes match videos, detects players and the ball, identifies the teams, determines player and ball positions on the football pitch, and classifies passes as successful or unsuccessful according to a predefined set of spatial rules.

The main objective is to determine whether a pass is successful or unsuccessful by analyzing the movement of the ball between players and the sequence of pitch cells through which the ball travels.

The system must not rely solely on whether the ball moves from one player to another. It must also consider the starting cell, target cell, intermediate cells, team relationships, and the predefined rules associated with these cells.

The project should be designed as a structured system with clearly separated detection, tracking, coordinate mapping, cell assignment, pass analysis, and result reporting components.

## 2. Objects and Information to Detect

The system must identify the following objects and information from football match videos.

### 2.1. Team A Players

Detect players belonging to Team A and identify their positions in each video frame.

The detected players must be tracked across consecutive frames so that their movement can be analyzed over time.

**Purpose:**
- Identify the players belonging to Team A.
- Determine their positions on the pitch.
- Identify potential pass receivers and passers.
- Analyze the movement of Team A players relative to the ball.
- Determine whether a pass reaches a teammate.

### 2.2. Team B Players

Detect players belonging to Team B and identify their positions in each video frame.

Team B players must also be tracked across consecutive frames.

**Purpose:**
- Identify opponents who may intercept a pass.
- Determine whether the ball reaches an opposing player.
- Analyze the relationship between Team A players, Team B players, and the ball.
- Support the classification of passes as successful or unsuccessful.

Team A and Team B identities must remain consistent throughout the analyzed sequence. A player's tracking identity and team identity are separate pieces of information and must be managed accordingly.

The system should also account for goalkeepers and referees where they can affect detection or tracking results. Referees must not be incorrectly classified as members of either team.

### 2.3. Ball Detection

Detect the football in each video frame and track its movement over time whenever possible.

The ball's position must be recorded independently of player positions.

**Purpose:**
- Determine the approximate starting position of a pass.
- Identify the player who appears to initiate a pass.
- Track the ball's trajectory.
- Identify the cells through which the ball travels.
- Determine whether the ball reaches a teammate, an opponent, or an area where possession cannot be established reliably.

Ball detection must account for temporary occlusion, motion blur, small ball size, and missed detections. A missing ball detection must not automatically be interpreted as a failed pass.

### 2.4. Football Pitch Lines and Key Points

Detect the visible football pitch lines and relevant pitch key points.

These may include pitch boundaries, penalty-area lines, goal-area lines, halfway lines, center-circle reference points, and other identifiable field markings.

**Purpose:**
- Establish the geometry of the football pitch.
- Determine the relationship between image coordinates and pitch coordinates.
- Map player and ball positions onto a consistent pitch coordinate system.
- Divide the pitch into a fixed grid of 50 cells.
- Ensure that the same real-world pitch location corresponds to the same cell throughout the analyzed sequence, even when the camera perspective changes.

Pitch lines and key points are used for spatial mapping. They do not independently determine whether a pass is successful or unsuccessful.

## 3. Pitch Coordinate Mapping

The system must transform the detected positions of players and the ball from image coordinates into pitch coordinates.

Image coordinates alone are insufficient because camera movement, perspective, and zoom can change the apparent positions and distances of objects.

The coordinate mapping process should establish a consistent representation of the pitch so that player positions, ball positions, and cell assignments can be compared meaningfully over time.

When the camera view changes, the system should update or re-establish the mapping as necessary.

If the pitch geometry cannot be estimated reliably for a particular frame, the system should mark the affected spatial results as uncertain rather than silently assigning incorrect coordinates.

## 4. Dividing the Pitch into 50 Cells

The football pitch must be divided into exactly 50 predefined spatial cells.

The grid is a core component of the pass analysis logic. Its purpose is to convert continuous player and ball positions into discrete spatial states that can be evaluated using explicit rules.

The system must:

- Define a consistent 50-cell layout covering the playable pitch.
- Assign each valid player position to a cell.
- Assign each valid ball position to a cell.
- Maintain stable cell identifiers throughout the analysis.
- Record the sequence of cells traversed by the ball during a potential pass.
- Identify the starting cell, intermediate cells, and target cell of each pass.
- Make cell assignments reproducible and consistent across frames.

The exact grid layout must be defined explicitly in the project implementation. Do not change the number of cells from 50 without authorization.

Cell boundaries and identifiers must be documented clearly so that pass rules can refer to specific cells unambiguously.

The grid must represent pitch locations rather than arbitrary image regions. A cell should continue to represent the same pitch area when the camera angle or zoom changes.

## 5. Pass Identification

The system must identify potential pass events by analyzing the relationship between players and the ball over time.

A potential pass should include, where the available evidence allows:

- The player who initiated the pass.
- The initiating player's team.
- The ball's starting position and starting cell.
- The sequence of intermediate cells traversed by the ball.
- The ball's target position and target cell.
- The player who receives the ball, if identifiable.
- The receiving player's team, if identifiable.
- The resulting pass classification.
- The confidence or uncertainty associated with the classification.

The system should distinguish a pass from other ball movements whenever possible, including dribbling, clearances, shots, rebounds, and deflections.

A sequence of ball detections alone does not prove that a pass occurred. The system must consider player proximity, changes in ball movement, possession transitions, and the available temporal evidence.

## 6. Rules for Classifying Passes

Every identified pass must be evaluated using a predefined rule-based process.

The classification logic must consider three spatial components:

1. **Starting cell:** The cell in which the pass begins.
2. **Intermediate cells:** The cells traversed by the ball during the pass.
3. **Target cell:** The cell in which the pass ends or the ball is received.

The rules must evaluate these components together with player and team identities.

### 6.1. Starting Cell

Determine the starting cell from the ball's position at the beginning of the pass.

Associate the pass with the player who appears to initiate it, when that player can be identified reliably.

The starting cell establishes the initial spatial state for evaluating the pass.

### 6.2. Intermediate Cells

Record the ordered sequence of cells traversed by the ball between the starting cell and the target cell.

Use this sequence to evaluate whether the ball's route satisfies the predefined pass rules.

Intermediate cells must be evaluated in their actual temporal order. A cell that the ball never entered must not be added to the trajectory merely because it lies between the starting and target cells geometrically.

The system must account for noisy detections and short-lived position errors without inventing unsupported cell transitions.

### 6.3. Target Cell

Determine the target cell from the ball's final relevant position or the location where possession is established.

Identify whether the ball reaches a teammate, an opponent, or no identifiable receiving player.

The target cell is essential for determining whether the intended pass reached an acceptable destination under the project's rules.

### 6.4. Successful Pass

Classify a pass as successful when the available evidence and predefined spatial rules establish that the pass satisfies the project's success conditions.

The evaluation must consider:

- The starting cell.
- The ordered sequence of intermediate cells.
- The target cell.
- The identity and team of the receiving player, when identifiable.
- Whether the ball reaches a teammate or otherwise satisfies the explicitly defined success rules.

A pass must not be classified as successful solely because the ball moved toward the intended target or passed through a particular cell.

### 6.5. Unsuccessful Pass

Classify a pass as unsuccessful when reliable evidence establishes that it violates the project's failure conditions.

Possible cases include:

- The ball is intercepted or controlled by an opposing player.
- The ball reaches an invalid destination under the predefined rules.
- The ball leaves the playable area.
- The ball fails to reach the intended teammate and the event can be reliably classified as a failed pass.
- The sequence of cells violates a rule explicitly defined for the pass.

These cases must be implemented according to the final rule definitions rather than treated as automatically equivalent in every circumstance.

### 6.6. Uncertain Events

The system must not force every ambiguous event into a successful or unsuccessful category.

If the ball is temporarily lost, the receiving player cannot be identified, or the available evidence is insufficient, the event should be marked as uncertain or requiring review.

Uncertain events must be reported separately and must not silently inflate the successful or unsuccessful pass counts.

## 7. Explicit Rule Configuration

The starting-cell, intermediate-cell, and target-cell rules must be represented explicitly in the implementation.

Do not bury the rules inside unrelated detection or tracking code.

The rule system should make it possible to define and inspect:

- Which starting cells are valid for a given rule.
- Which intermediate-cell sequences are permitted or prohibited.
- Which target cells are acceptable.
- Which team and possession conditions are required.
- Which conditions lead to a successful, unsuccessful, or uncertain result.

The exact cell-to-cell rules must be documented and configurable. Do not invent additional football tactics or success criteria without authorization.

The system must distinguish between detecting what happened in the video and evaluating what happened against the project's rules.

Detection and tracking provide observations. The rule engine uses those observations to determine the classification.

## 8. Pass Records and Results

For every analyzed pass, store a structured record containing, where available:

- Pass identifier.
- Start and end timestamps.
- Passing player identifier.
- Receiving player identifier.
- Team identifier.
- Starting cell.
- Ordered intermediate-cell sequence.
- Target cell.
- Relevant player and ball positions.
- Pass classification: successful, unsuccessful, or uncertain.
- Reason for the classification.
- Relevant confidence or data-quality indicators.

The system should also produce summary statistics, including:

- Total identified passes.
- Successful passes.
- Unsuccessful passes.
- Uncertain passes.
- Pass counts by team.
- Successful and unsuccessful pass rates for each team.

The summary must use clearly defined counting rules so that a single pass is not counted multiple times due to repeated detections across video frames.

## 9. Project Architecture and Development Principles

Organize the project into clearly separated responsibilities:

1. Video input and frame processing.
2. Player, team, ball, pitch-line, and key-point detection.
3. Player and ball tracking.
4. Pitch coordinate mapping.
5. Fifty-cell grid representation.
6. Pass event identification.
7. Rule-based pass classification.
8. Result storage, statistics, and visualization.

These are logical responsibilities, not mandatory technology choices.

Each component must have clearly defined inputs and outputs so that it can be tested independently.

The implementation should be developed incrementally. Validate object detections and spatial mappings before relying on them for pass classification.

Keep detection failures, tracking errors, coordinate-mapping errors, and rule violations distinguishable during debugging.

Do not claim that the system correctly classifies passes until it has been evaluated against manually verified examples.

## 10. Expected Project Outcome

The completed system should be able to analyze a football match video and:

1. Detect Team A players, Team B players, the ball, pitch lines, and pitch key points.
2. Track players and the ball over time.
3. map player and ball positions into consistent pitch coordinates.
4. Assign positions to the appropriate cells in the 50-cell pitch grid.
5. Identify potential passes and extract their starting, intermediate, and target cells.
6. Evaluate each pass using the explicitly defined rules.
7. Classify each pass as successful, unsuccessful, or uncertain.
8. Produce pass counts, team-level statistics, and traceable explanations for classifications.

The primary objective is not merely to detect football objects. It is to use reliable object detection, tracking, spatial mapping, and explicit cell-based rules to analyze football passes consistently and transparently.

## 11. Instructions for the Initial Implementation

Begin by translating this specification into a clear project structure and an implementation plan.

Identify the required modules, their responsibilities, their inputs and outputs, and the order in which they should be developed and tested.

Keep the project scope aligned with this specification. Do not add unrelated features.

Do not silently invent missing pass rules. Where a rule has not yet been specified, document it as an unresolved rule that requires an explicit definition before that rule can be considered final.

Prioritize a small, verifiable end-to-end prototype before expanding the system.

The first milestone should demonstrate the complete logical flow on a short video segment: object observations, tracking, pitch mapping, 50-cell assignment, pass-event analysis, rule-based classification, and a traceable result record.

All implementation decisions must support the central objective: **classifying football passes as successful or unsuccessful based on player and ball movement through the 50-cell pitch grid and the explicitly defined project rules.**
