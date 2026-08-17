# Canonical Interpretability Trace (3 Cases)

The traces use the canonical four-layer trees, SW-BTED structural-only cost parameters, and APTED mappings.

## Case A: `SU26SE102–SU26SE102_plag`
Role: positive plagiarism pair; label=1.

Nodes: 86 vs 41; structural similarity: `0.3656`.

| Domain | Similarity | Replacements | Deletes | Inserts |
|---|---:|---:|---:|---:|
| D1_BUSINESS_CONTEXT | 0.4209 | 7 | 19 | 1 |
| D2_FUNCTIONAL | 0.4650 | 1 | 22 | 4 |
| D3_TECHNICAL_REALIZATION | 0.5038 | 9 | 9 | 0 |
| D4_EXECUTION_PLANNING | 0.6307 | 9 | 0 | 0 |

Representative replacements:
- D1_BUSINESS_CONTEXT (T3): `IQGS – AI-Powered Interview Question Generation System Using RAG and LLM` → `IQGS – AI-Powered Interview Question Generation System Using RAG and LLM IQGS – Hệ thống Sinh Câu hỏi Phỏng vấn sử dụng RAG và LLM The contemporary recruitment landscape is characterized by a significant demand for high quality, role specific interview questions, a task that often burdens human resources professionals and interviewers. In response to this growing challenge, over 60% of enterprises resort to leveraging outdated question compilations or rely heavily on the individualized approache`
- D1_BUSINESS_CONTEXT (T4): `interview question` → `interview`
- D2_FUNCTIONAL (T3): `1. User Roles and Access Control 1.1 User Roles The system shall support three primary user roles: 	Administrator ● Manage users and roles ● Manage knowledge base and system configuration ● Monitor system performance, analytics, and transactions 	HR Manager / Recruiter ● Input job descriptions ● Generate interview questions using AI (RAG-based) ● Review, edit, and publish question sets ● View analytics and candidate interactions ● Manage subscription and feature access 	Job Seeker / Candidate` → `The proposed system, referred to as the Intelligent Interview Framework (IIF), will accommodate three fundamental user roles: The Administrator will be responsible for user and role management , overseeing the knowledge repository and system settings , 1.1 User Roles tracking system performance , analytics , and transactions . HR Manager / Recruiter The HR Manager or Recruiter will be able to input job descriptions , utilize artificial intelligence to formulate interview questions based on Retri`
- D3_TECHNICAL_REALIZATION (T3): `● Usability: The system shall provide an intuitive interface ● Maintainability: The system shall be modular and easy to extend (*) 3.2. Main proposal content (including result and product)` → `The proposed framework, dubbed QueryConstruct, leverages the power of Retrieval Augmented Generation (RAG) in conjunction with Large Language Models (LLMs) to facilitate the automation of question generation tasks. This approach is further enhanced by the application of advanced Natural Language Processing (NLP) methodologies, which serve pivotal roles in document analysis, the extraction of pertinent information, and enabling semantic search capabilities. 4. Detailed Design Document In addition`
- D3_TECHNICAL_REALIZATION (T4): `interface maintainability` → `queryconstruct`
- D4_EXECUTION_PLANNING (T3): `● Task 1: Requirement analysis, knowledge base source collection and system architecture design. ● Task 2: Knowledge base preparation (document ingestion, chunking, embedding, vector indexing) and database setup. ● Task 3: RAG pipeline implementation (embedding, semantic retrieval, LLM generation, output validation, async background job infrastructure). ● Task 4: Backend REST API development — all endpoints for HR, Job Seeker, marketplace, analytics, and RAG evaluation experiment. ● Task 5: HR p` → `Phase 1: Needs Assessment and System Blueprint Development The initial phase will focus on the identification of user requirements and the collection of relevant knowledge resources, culminating in a comprehensive architectural design for the system, referred to as JobLinker. Sprint 2: Knowledge Resource Configuration and Database Initialization This stage will encompass the preparation of the knowledge base through processes such as document ingestion, segmentation, embedding, and the establish`
- D4_EXECUTION_PLANNING (T4): `evaluation experiment` → `job seekers`

## Case B: `SP26SE068–SU26SE063`
Role: SBERT false positive; SW-BTED correct; label=0.

Nodes: 86 vs 86; structural similarity: `0.2818`.

| Domain | Similarity | Replacements | Deletes | Inserts |
|---|---:|---:|---:|---:|
| D1_BUSINESS_CONTEXT | 0.3656 | 27 | 0 | 0 |
| D2_FUNCTIONAL | 0.3753 | 27 | 0 | 0 |
| D3_TECHNICAL_REALIZATION | 0.4600 | 18 | 0 | 0 |
| D4_EXECUTION_PLANNING | 0.6009 | 9 | 0 | 0 |

Representative replacements:
- D1_BUSINESS_CONTEXT (T3): `In the context of the increasing demand for organizing home parties, company parties, and internal events, individuals and businesses often have difficulty finding reputable service providers, comparing prices, and booking services quickly. Many service providers (restaurants, mobile catering, event organization services) do not have a convenient online channel to reach customers.
The Home and Company Party Booking Platform project was built to connect customers with party services in a convenie` → `Board game cafes currently face difficulties in managing revenue across complex time slots and controlling the risk of losing thousands of small game components. For players, they often struggle with a lack of teammates and difficulty finding groups with similar interests and skill levels (Elo) in their vicinity. These issues lead to operational inefficiencies for cafe owners and a fragmented experience for the board game community.`
- D1_BUSINESS_CONTEXT (T4): `services convenient` → `board game`
- D2_FUNCTIONAL (T3): `o	Web application for Owner
	Owner role:
•	Login/Logout
•	Manage user accounts (enable/disable, view list)
•	Provide all kinds of services for buffet parties
•	View raw material inventory information
•	Create groups by job type
•	Assign order to groups
•	Manage Order
•	Chat with customer
•	View order history
•	View Feedback
•	View statistics 
o	Mobile & Web application for Group Leader, Staff and Customer
	Group Leader role:
•	Login/Logout
•	View orders and create corresponding work lists
•	As` → `Player Mobile App: ● Authentication: Users can register, login, and manage personal profiles (interests, Elo rating, Karma points). ● Discovery: Users can search for partner cafes, view real-time table availability, and browse the available game list of each cafe. ● Matchmaking: Users can create a Lobby with specific conditions: game type, search radius, minimum skill level, and desired time slot. ● Notifications: The system automatically suggests and sends Push Notifications to suitable players`
- D2_FUNCTIONAL (T4): `role login` → `cafe`
- D3_TECHNICAL_REALIZATION (T3): `•	The system shall support at least 100 concurrent users with an average response time under 2 seconds per request.
•	GUI must be simple, friendly, and easy to use.
•	The system must be scalable to a chain of stores.
•	The language used in the application is Vietnamese.
•	Web applications send requests to the server through HTTPS protocol.
•	The recommendation system will be implemented using the best suitable for lower setting server.` → `● Performance: The system API response time must be less than 500ms. ● Availability: The system guarantees 99.9% uptime with an auto-failover mechanism. ● Security: All sensitive data and user passwords must be encrypted using high-security hashing algorithms. ● Usability: The POS and Management Web apps must be responsive, optimizing operations on Tablets and Mobile devices. (*) 3.2. Main proposal content (including result and product)`
- D3_TECHNICAL_REALIZATION (T4): `scalable chain` → `usability pos`
- D4_EXECUTION_PLANNING (T3): `o	Task package 1: Design UI Elements for the Web Application
o	Task package 2: Develop API for the System
o	Task package 3: Develop the Web Application
o	Task package 4: Develop the Mobile Application
o	Task package 5: Build, Deploy, and Test the System
o	Task package 6: Prepare Required Documents:
o	System Analysis and Design
o	Test Plan
o	Installation Manual
o	User Manual
o	Each work group may have many students participating but there will be one member responsible for the main responsibility` → `● Task 1: Develop the Web applications for Admin, Cafe Partners, and Event Organizers. ● Task 2: Develop the Core API for Matchmaking, Dynamic Inventory, and Billing systems. ● Task 3: Develop the Mobile App for Players. ● Task 4: Build, Deploy, and Test the entire system. ● Task 5: Prepare technical documents: System Analysis and Design, Test Plan, Installation Manual, and User Manual.`
- D4_EXECUTION_PLANNING (T4): `application task` → `task develop`

## Case C: `SP26SE122–SP26SE055`
Role: negative Type_C pair; both correct; label=0.

Nodes: 86 vs 86; structural similarity: `0.2916`.

| Domain | Similarity | Replacements | Deletes | Inserts |
|---|---:|---:|---:|---:|
| D1_BUSINESS_CONTEXT | 0.3664 | 27 | 0 | 0 |
| D2_FUNCTIONAL | 0.3908 | 27 | 0 | 0 |
| D3_TECHNICAL_REALIZATION | 0.4687 | 18 | 0 | 0 |
| D4_EXECUTION_PLANNING | 0.6013 | 9 | 0 | 0 |

Representative replacements:
- D1_BUSINESS_CONTEXT (T3): `This project focuses on the development of a Manufacturing Execution System (MES) module, specialized for companies operating in the Paper Packaging and Offset Printing Industry. The core objective is to automate the entire production management lifecycle in a Make-to-Order (MTO) environment, dramatically improving planning efficiency and material control.

The system is designed to transform complex, multi-stage workflows—including ten distinct production steps (e.g., Ralo/Slitting, Printing, L` → `The rapid expansion of urban populations continues to increase the demand for long-term rental housing such as mini apartments and serviced residential units. Modern tenants, especially students and young professionals, expect a professional, transparent, and standardized living experience—including accurate billing, timely maintenance, stable contracts, and predictable living costs.

However, the majority of landlords still rely on manual or semi-digital management through spreadsheets, messagi`
- D1_BUSINESS_CONTEXT (T4): `printing` → `pain points`
- D2_FUNCTIONAL (T3): `Based on user requirements, the system is implemented as follows:
-	Development of Customer & Sales Application: Allows Customers to submit specifications for quotations and track order status in real time; allows Sales staff to manage requests and fulfill orders.
-	Build order approval management interface: Real-time approves customer orders and raw material purchase requests from the automated system.
-	Implementation of Automated MRP Event: System handler to automatically analyze Bill of Mate` → `Tenant (Mobile App)
Account & Profile
Register, log in/out
Update personal profile
Rental Management
Search and view available serviced apartments
Submit rental applications
View contract details and lifecycle
Request contract extension or termination
Billing & Payments
View rental and service invoices
Pay online
See transaction history
Receive due/overdue reminders
Maintenance
Create maintenance requests with images
Track repair progress
Review and rate completed work
IoT Utility Usage
View rea`
- D2_FUNCTIONAL (T4): `purchase requests` → `tenant`
- D3_TECHNICAL_REALIZATION (T3): `-	Some limitations are:
●	Multi-plant Support: The system is designed for a single factory location and does not support multi-site inventory synchronization or inter-branch transfers.
●	Static Scheduling Only: The production schedule is based on fixed capacity planning. It does not support dynamic, AI-driven rescheduling in real-time based on unexpected machine breakdowns or sensor data.
●	Estimated Wastage Calculation: Material wastage is calculated based on a fixed percentage or predefined ru` → `Supports multiple digital payment methods
Mobile app available on iOS and Android
Web application compatible across major browsers and devices
Real-time alerts and monitoring for critical operations
Scalable architecture capable of handling multiple properties and high data volume`
- D3_TECHNICAL_REALIZATION (T4): `does support` → `methods mobile`
- D4_EXECUTION_PLANNING (T3): `-	The first student: Should implement the Base Architecture using .NET Core, Database Design (SQL Server, MySQL, ...), Authentication & Authorization, and Master Data Management APIs. This student is also responsible for integrating third-party services for file storage.
-	The second student: Should implement the complex MRP Logic APIs (BOM explosion, Wastage calculation), the Task Assignment and Notification systems, and the System Handler background services for auto-triggering Inventory Check` → `Requirement Analysis & UI/UX Design
Database & System Architecture Design
Web Application Development
Mobile Application Development
IoT Integration Module
System Testing (Unit, Integration, UAT)
Deployment & Release
Documentation & Final Presentation`
- D4_EXECUTION_PLANNING (T4): `meeting client` → `development iot`
